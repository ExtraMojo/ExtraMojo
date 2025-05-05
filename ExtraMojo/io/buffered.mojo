"""
Buffered reading and writing.

## Examples

BufferedReader:

```mojo
from testing import assert_equal
from ExtraMojo.io.buffered import BufferedReader

fn test_read_until(file: String, expected_lines: List[String]) raises:
    var buffer_capacities = List(10, 100, 200, 500)
    for cap in buffer_capacities:
        var fh = open(file, "r")
        var reader = BufferedReader(fh^, buffer_capacity=cap[])
        var buffer = List[UInt8]()
        var counter = 0
        while reader.read_until(buffer) != 0:
            assert_equal(List(expected_lines[counter].as_bytes()), buffer)
            counter += 1
            buffer.clear()
        assert_equal(counter, len(expected_lines))
        print("Successful read_until with buffer capacity of {}".format(cap[]))

```

BufferedWriter:

```mojo
from ExtraMojo.io.buffered import BufferedWriter

fn test_buffered_writer(file: String, expected_lines: List[String]) raises:
    var fh = BufferedWriter(open(String(file), "w"), buffer_capacity=128)
    for i in range(len(expected_lines)):
        fh.write_bytes(expected_lines[i].as_bytes())
        fh.write_bytes("\n".as_bytes())
    fh.flush()
    fh.close()

    test_read_until(String(file), expected_lines)

```

"""
import math
from algorithm import vectorize
from memory import UnsafePointer, memcpy
from sys.info import simdwidthof
from utils import Writable


from ExtraMojo.io import MovableWriter
from ExtraMojo.bstr.bstr import (
    find_chr_all_occurrences,
)
from ExtraMojo.bstr.memchr import memchr


alias NEW_LINE = 10
alias SIMD_U8_WIDTH: Int = simdwidthof[DType.uint8]()
# 128 KiB: http://git.savannah.gnu.org/gitweb/?p=coreutils.git;a=blob;f=src/ioblksize.h;h=266c209f48fc07cb4527139a2548b6398b75f740;hb=HEAD#l23
alias BUF_SIZE: Int = 1024 * 128


fn read_lines(
    path: String, buf_size: Int = BUF_SIZE
) raises -> List[List[UInt8]]:
    """
    Read all the lines in the file and return them as a List of Lists of bytes.

    Args:
        path: The file path to open and read.
        buf_size: The size of the buffer to use.

    Returns:
        A list of lines, where each line is a `List[UInt8]`.
    """
    # TODO: make this an iterator
    var fh = open(path, "r")
    var result = List[List[UInt8]]()
    var file_pos = 0

    while True:
        _ = fh.seek(file_pos)
        var buffer = fh.read_bytes(buf_size)
        var newlines = find_chr_all_occurrences(buffer, NEW_LINE)
        var start = 0
        for i in range(0, len(newlines)):
            var newline = newlines[i]
            result.append(buffer[start:newline])
            start = newline + 1

        if len(buffer) < BUF_SIZE:
            break
        file_pos += start
    return result


fn for_each_line[
    func: fn (Span[UInt8], Int, Int) capturing -> None
](path: String, buf_size: Int = BUF_SIZE) raises:
    """
    Call the provided callback on each line.

    The callback will be given a buffer, and the [start, end) of where the line is in that buffer.

    Parameters:
        func: The callback to run for each line.

    Args:
        path: The file path to open and read.
        buf_size: The size of the buffer to use.

    """
    var fh = open(path, "r")
    # var result = List[Tensor[DType.int8]]()
    var file_pos = 0

    while True:
        _ = fh.seek(file_pos)
        var buffer = fh.read_bytes(buf_size)
        var buffer_index = 0

        while True:
            var newline = memchr(buffer, NEW_LINE, buffer_index)
            if newline == -1:
                break

            func(buffer, buffer_index, newline)
            buffer_index = newline + 1

        file_pos += buffer_index
        if len(buffer) < BUF_SIZE:
            break


@always_inline
fn get_next_line[
    is_mutable: Bool, //, origin: Origin[is_mutable]
](buffer: Span[UInt8, origin], start: Int) -> Span[UInt8, origin]:
    """Function to get the next line using either SIMD instruction (default) or iteratively.

    Args:
        buffer: The buffer to search for the next newline character.
        start: The start position to use inside the buffer.

    Returns:
        A span of bytes from [start, newline).
    """

    var in_start = start
    while buffer[in_start] == NEW_LINE:  # Skip leading \n
        in_start += 1
        if in_start >= len(buffer):
            return buffer[0:0]

    var next_line_pos = memchr(buffer, NEW_LINE, in_start)
    if next_line_pos == -1:
        next_line_pos = len(
            buffer
        )  # If no line separator found, return the reminder of the string, behavior subject to change
    return buffer[in_start:next_line_pos]


struct BufferedReader:
    """
    BufferedReader for readying lines and bytes from a file in a buffered way.

    ## Example

    ```mojo
    from ExtraMojo.io.buffered import BufferedReader

    fn read_bytes(read file: String) raises -> List[UInt8]:
        var fh = open(file, "r")
        var reader = BufferedReader(fh^, buffer_capacity=50)
        var buffer = List[UInt8](capacity=125)
        for _ in range(0, 125):
            buffer.append(0)
        var found_file = List[UInt8]()

        # Read bytes from the buf reader, copy to found
        var bytes_read = 0
        while True:
            bytes_read = reader.read_bytes(buffer)
            if bytes_read == 0:
                break
            found_file.extend(buffer[0:bytes_read])
        return found_file
    ```
    """

    var fh: FileHandle
    """The internal filehandle to read from."""
    var buffer: UnsafePointer[UInt8]
    """The internal buffer."""
    var file_offset: Int
    """Current offset into the file."""
    var buffer_offset: Int
    """Current offset into the buffer."""
    var buffer_capacity: Int
    """Total capacity of the buffer."""
    var buffer_len: Int
    """Total filled capacity of the buffer."""

    fn __init__(
        out self, owned fh: FileHandle, buffer_capacity: Int = BUF_SIZE
    ) raises:
        """Create a `BufferedReader`.

        Args:
            fh: The filehandle to read from.
            buffer_capacity: The size of the buffer to use.
        """
        self.fh = fh^
        self.file_offset = 0
        self.buffer_offset = 0
        self.buffer_capacity = buffer_capacity
        self.buffer = UnsafePointer[UInt8].alloc(self.buffer_capacity)
        self.buffer_len = 0
        _ = self._fill_buffer()

    fn __del__(owned self):
        try:
            self.fh.close()
        except:
            pass
        self.buffer.free()

    fn __enter__(owned self) -> Self:
        return self^

    fn __moveinit__(out self, owned existing: Self):
        self.fh = existing.fh^
        self.file_offset = existing.file_offset
        self.buffer_offset = existing.buffer_offset
        self.buffer = existing.buffer
        self.buffer_capacity = existing.buffer_capacity
        self.buffer_len = existing.buffer_len

    fn read_bytes(mut self, mut buffer: List[UInt8]) raises -> Int:
        """Read up to `len(buffer)` bytes.

        Args:
            buffer: The buffer to read into. The `len` of the `buffer` determines how many bytes will be read.

        Returns:
            This returns the number of bytes read.
            If the number of bytes read is less then `len(buffer)` then EOF has been reached.
        """
        var bytes_to_read = len(buffer)
        var bytes_read = 0

        while bytes_to_read > 0:
            var out_buf_ptr = buffer.unsafe_ptr().offset(bytes_read)
            # Copy as much as possible into the buffer
            var available_bytes = min(
                self.buffer_len - self.buffer_offset, bytes_to_read
            )
            memcpy(
                out_buf_ptr,
                self.buffer.offset(self.buffer_offset),
                available_bytes,
            )
            self.buffer_offset += available_bytes
            bytes_to_read -= available_bytes
            bytes_read += available_bytes

            if self.buffer_offset == self.buffer_len:
                var bytes_filled = self._fill_buffer()
                if bytes_filled == 0:
                    return bytes_read

        return bytes_read

    fn read_until(
        mut self,
        mut buffer: List[UInt8],
        char: UInt = NEW_LINE,
    ) raises -> Int:
        """
        Fill the given `line_buffer` until the given `char` is hit, or EOF.

        **Note**: the callee is responsible for clearing (or not) the buffer between calls to `read_until`.

        Args:
            buffer: The buffer to filled with any bytes found before `char` is hit.
            char: The character to use as the terminator.

        Returns:
            The number of bytes read.
        """

        var bytes_read = 0

        while True:
            # Find the next newline in the buffer
            var newline_index = memchr(
                Span[UInt8, __origin_of(self)](
                    ptr=self.buffer, length=self.buffer_len
                ),
                char,
                self.buffer_offset,
            )

            # Copy the line into the provided buffer, if there was no newline, copy in the remainder of the buffer
            var end = newline_index if newline_index != -1 else self.buffer_len
            var size = end - self.buffer_offset
            buffer.reserve(len(buffer) + size)
            var line_ptr = buffer.unsafe_ptr().offset(len(buffer))
            memcpy(line_ptr, self.buffer.offset(self.buffer_offset), size)
            # TODO: is there a better way to do this?
            buffer._len += size

            # Advance our position in our buffer
            self.buffer_offset = newline_index + 1
            bytes_read = len(buffer) + newline_index + 1

            # Try to refill the buffer
            if newline_index == -1:
                self.file_offset += self.buffer_offset
                var bytes_filled = self._fill_buffer()
                if bytes_filled == 0:
                    break
            else:
                break

        return bytes_read

    fn _fill_buffer(mut self) raises -> Int:
        """Fill the buffer, dropping anything currently not read.

        Returns:
            The number of bytes read
        """
        var buf_ptr = self.buffer
        var bytes_read = self.fh.read(
            Span[UInt8, __origin_of(buf_ptr)](
                ptr=buf_ptr, length=self.buffer_capacity
            )
        )
        self.buffer_len = bytes_read.__int__()
        self.buffer_offset = 0
        return self.buffer_len


struct BufferedWriter[W: MovableWriter](Writer):
    """A BufferedWriter.

    ## Example

    ```mojo
    from ExtraMojo.io.buffered import BufferedWriter
    fn write_to_file(read file: String, read expected_lines: List[String]) raises:
        var fh = BufferedWriter(open(String(file), "w"), buffer_capacity=128)
        for i in range(len(expected_lines)):
            fh.write_bytes(expected_lines[i].as_bytes())
            fh.write_bytes("\n".as_bytes())
        fh.flush()
        fh.close()
    ```
    """

    var inner: W
    """The inner file handle to write to."""
    var buffer: List[UInt8]
    """The inner buffer."""
    var buffer_capacity: Int
    """The capacity of the inner buffer."""
    var buffer_len: Int
    """The number of bytes currently stored in the inner buffer."""

    fn __init__(
        out self, owned writer: W, buffer_capacity: Int = BUF_SIZE
    ) raises:
        """Create a `BufferedReader`.

        Args:
            writer: The writer to write to.
            buffer_capacity: The capacity of the inner buffer to use.
        """
        self.inner = writer^
        self.buffer = List[UInt8](capacity=buffer_capacity)
        self.buffer_capacity = buffer_capacity
        self.buffer_len = 0

    fn __del__(owned self):
        self.flush()

    fn __enter__(owned self) -> Self:
        return self^

    fn __moveinit__(out self, owned existing: Self):
        self.inner = existing.inner^
        self.buffer = existing.buffer^
        self.buffer_capacity = existing.buffer_capacity
        self.buffer_len = existing.buffer_len

    fn close(mut self) raises:
        self.flush()
        # self.fh.close()

    fn write_bytes(mut self, bytes: Span[UInt8]):
        """Write bytes to this writer.

        Args:
            bytes: The bytes that will be written to the underlying buffer.
        """
        var b = bytes
        while True:
            var end = min(self.buffer_capacity - self.buffer_len, len(b))

            var to_copy = b[:end]
            memcpy(
                self.buffer.unsafe_ptr().offset(self.buffer_len),
                to_copy.unsafe_ptr(),
                len(to_copy),
            )
            self.buffer._len += len(to_copy)
            self.buffer_len += len(to_copy)
            if len(to_copy) == len(b):
                break
            else:
                self.flush()
            b = b[end:]

    fn write[*Ts: Writable](mut self, *args: *Ts):
        """Implement write.

        Args:
            args: Any `Writable` values that will be written to the writer.
        """

        @parameter
        fn write_arg[T: Writable](arg: T):
            arg.write_to(self)

        args.each[write_arg]()

    fn flush(mut self):
        """Write any remaining bytes in the current buffer, then clear the buffer.
        """
        self.inner.write_bytes(Span(self.buffer))
        self.buffer_len = 0
        self.buffer.clear()
