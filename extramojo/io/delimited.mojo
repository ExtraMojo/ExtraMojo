"""Working with simple delimited text.

## Example

Compile-time known fields:

TODO: this should be two different examples, but the doc parser can't seem to handle that for this example.

```mojo
from collections.string import StringSlice
from testing import assert_equal

from extramojo.bstr.bstr import SplitIterator
from extramojo.cli.parser import ParsedOpts
from extramojo.io.buffered import (
    BufferedReader,
    BufferedWriter,
)
from extramojo.io.delimited import (
    DelimReader,
    FromDelimited,
    ToDelimited,
    DelimWriter,
)

# #########################################
# Example with compile-time known fields.
# #########################################

@fieldwise_init
struct SerDerStruct(ToDelimited, FromDelimited):
    var index: Int
    var name: String

    fn write_to_delimited(read self, mut writer: DelimWriter) raises:
        writer.write_record(self.index, self.name)

    fn write_header(read self, mut writer: DelimWriter) raises:
        writer.write_record("index", "name")

    @staticmethod
    fn from_delimited(mut data: SplitIterator, read header_values: Optional[List[String]]=None) raises -> Self:
        var index = Int(StringSlice(unsafe_from_utf8=data.__next__()))
        var name = String()  # String constructor expected nul terminated byte span
        name.write_bytes(data.__next__())
        return Self(index, name)


fn test_delim_reader_writer(file: String) raises:
    var to_write = List[SerDerStruct]()
    for i in range(0, 1000):
        to_write.append(SerDerStruct(i, String("MyNameIs" + String(i))))
    var writer = DelimWriter(
        BufferedWriter(open(String(file), "w")), delim="\t", write_header=True
    )
    for item in to_write:
        writer.serialize(item)
    writer.flush()
    writer.close()

    var reader = DelimReader[SerDerStruct](
        BufferedReader(open(String(file), "r")),
        delim=ord("\t"),
        has_header=True,
    )
    var count = 0
    for item in reader^:
        assert_equal(to_write[count].index, item.index)
        assert_equal(to_write[count].name, item.name)
        count += 1
    assert_equal(count, len(to_write))

# #########################################
# Example with dynamic fields.
# #########################################

@value
struct Score[
    truth_lengths_origin: ImmutableOrigin,
    truth_names_origin: ImmutableOrigin,
](ToDelimited):
    var assembly_name: String
    var assembly_length: Int
    var scores: List[Int32]
    var truth_lengths: Pointer[List[Int], truth_lengths_origin]
    var truth_names: Pointer[List[String], truth_names_origin]

    fn __init__(
        out self,
        var assembly_name: String,
        assembly_length: Int,
        var scores: List[Int32],
        ref [truth_lengths_origin]truth_lengths: List[Int],
        ref [truth_names_origin]truth_names: List[String],
    ):
        self.assembly_name = assembly_name^
        self.assembly_length = assembly_length
        self.scores = scores^
        self.truth_lengths = Pointer(to=truth_lengths)
        self.truth_names = Pointer(to=truth_names)

    fn write_to_delimited(read self, mut writer: DelimWriter) raises:
        writer.write_field(self.assembly_name, is_last=False)
        writer.write_field(self.assembly_length, is_last=False)
        for i in range(0, len(self.scores)):
            writer.write_field(
                String("{}/{}").format(self.scores[i], self.truth_lengths[][i]),
                is_last=i == len(self.scores) - 1,
            )

    fn write_header(read self, mut writer: DelimWriter) raises:
        writer.write_field("assembly_name", is_last=False)
        writer.write_field("assembly_length", is_last=False)
        for i in range(0, len(self.truth_names[])):
            writer.write_field(
                self.truth_names[][i], is_last=i == len(self.truth_names[]) - 1
            )

fn run_check_scores(opts: ParsedOpts) raises:
    var truth_names = List(String("A"), String("B"), String("C"))
    var truth_lengths = List(125, 2000, 1234)
    var output_scores_tsv = "/tmp/out.tsv"

    var scores = List(
        Score(String("Assembly1"), 100, List[Int32](1, 2, 3), truth_lengths, truth_names),
        Score(String("Assembly2"), 100, List[Int32](100, 2, 3), truth_lengths, truth_names),
        Score(String("Assembly3"), 100, List[Int32](1, 100, 3), truth_lengths, truth_names),
        Score(String("Assembly4"), 100, List[Int32](1, 2, 100), truth_lengths, truth_names)
    )

    var out_writer = DelimWriter(
        BufferedWriter(open(output_scores_tsv, "w")),
        delim="\t",
        write_header=True,
    )

    for score in scores:
        out_writer.serialize[
            Score[__origin_of(truth_lengths), __origin_of(truth_names)]
        ](score)

    out_writer.flush()
    out_writer.close()

# #########################################
# Example similar to dictreader/dictwriter.
# #########################################

@value
struct ThinWrapper(ToDelimited, FromDelimited):
    var stuff: Dict[String, Int]

    fn write_to_delimited(read self, mut writer: DelimWriter) raises:
        var seen = 1
        for value in self.stuff.values():  # Relying on stable iteration order
            writer.write_field(value, is_last=seen == len(self.stuff))
            seen += 1

    fn write_header(read self, mut writer: DelimWriter) raises:
        var seen = 1
        for header in self.stuff.keys():  # Relying on stable iteration order
            writer.write_field(header, is_last=seen == len(self.stuff))
            seen += 1

    @staticmethod
    fn from_delimited(
        mut data: SplitIterator,
        read header_values: Optional[List[String]] = None,
    ) raises -> Self:
        var result = Dict[String, Int]()
        for header in header_values.value():
            result[header] = Int(
                StringSlice(unsafe_from_utf8=data.__next__())
            )
        return Self(result)


fn test_delim_reader_writer_dicts(file: String) raises:
    var to_write = List[ThinWrapper]()
    var headers = List(
        String("a"), String("b"), String("c"), String("d"), String("e")
    )
    for i in range(0, 1000):
        var stuff = Dict[String, Int]()
        for header in headers:
            stuff[header] = i
        to_write.append(ThinWrapper(stuff))
    var writer = DelimWriter(
        BufferedWriter(open(String(file), "w")),
        delim="\t",
        write_header=True,
    )
    for item in to_write:
        writer.serialize(item)
    writer.flush()
    writer.close()

    var reader = DelimReader[ThinWrapper](
        BufferedReader(open(String(file), "r")),
        delim=ord("\t"),
        has_header=True,
    )
    var count = 0
    for item in reader^:
        for header in headers:
            assert_equal(to_write[count].stuff[header], item.stuff[header])
        count += 1
    assert_equal(count, len(to_write))
    print("Successful delim_writer")
```
"""
from io import Writer

from extramojo.bstr.bstr import SplitIterator
from extramojo.io.buffered import BufferedReader, BufferedWriter


trait FromDelimited(Copyable & Movable):
    """Create an instance of `Self` from the iterator over `Span[UInt8]` bytes.
    """

    @staticmethod
    fn from_delimited(
        mut data: SplitIterator,
        read header_values: Optional[List[String]] = None,
    ) raises -> Self:
        ...


struct DelimReader[RowType: FromDelimited](Movable):
    """Read delimited data that is delimited by a single bytes.

    The `RowType` must implement `FromBytes` which is passed an iterator over the split up line.
    """

    # TODO: there's something a bit odd about how this all works as an iterator.
    # I can't add a context manager, and the iterator has to use `^` which I'm not sure about.

    var delim: UInt8
    var reader: BufferedReader
    var next_elem: Optional[Self.RowType]
    var buffer: List[UInt8]
    var len: Int
    var has_header: Bool
    var header_values: Optional[List[String]]

    fn __init__(
        out self,
        var reader: BufferedReader,
        *,
        delim: UInt8,
        has_header: Bool,
    ) raises:
        self.delim = delim
        self.reader = reader^
        self.next_elem = None
        self.buffer = List[UInt8]()
        self.len = 1
        self.has_header = has_header
        self.header_values = None
        if self.has_header:
            self._skip_header()
        self._get_next()

    fn __moveinit__(out self, deinit existing: Self):
        self.delim = existing.delim
        self.reader = existing.reader^
        self.next_elem = existing.next_elem^
        self.buffer = existing.buffer^
        self.len = existing.len
        self.has_header = existing.has_header
        self.header_values = existing.header_values^

    fn __len__(read self) -> Int:
        return self.len

    fn __has_next__(read self) -> Bool:
        return self.__len__() > 0

    fn __next__(mut self) raises -> Self.RowType:
        if not self.next_elem:
            raise "Attempting to call past end of iterator"
        var ret = self.next_elem.take()
        self._get_next()
        return ret^

    fn __iter__(var self) -> Self:
        return self^

    fn _skip_header(mut self) raises:
        self.buffer.clear()
        var bytes_read = self.reader.read_until(self.buffer, UInt(ord("\n")))

        if bytes_read == 0:
            self.len = 0
            self.next_elem = None
            raise "No header found"

        var header_values = List[String]()
        for header in SplitIterator(self.buffer, self.delim):
            header_values.append(String(StringSlice(unsafe_from_utf8=header)))
        self.header_values = header_values^

    fn _get_next(mut self) raises:
        self.buffer.clear()
        var bytes_read = self.reader.read_until(self.buffer, UInt(ord("\n")))
        var tmp = String()
        tmp.write_bytes(self.buffer)
        if bytes_read == 0:
            self.len = 0
            self.next_elem = None
            return
        var iterator = SplitIterator(self.buffer, self.delim)
        self.next_elem = Self.RowType.from_delimited(
            iterator, self.header_values
        )


trait ToDelimited:
    fn write_to_delimited(read self, mut writer: DelimWriter) raises:
        """Write `self` to the passed in `writer`.

        This should probably be done with `DelimWriter.write_record` or a series of
        `DelimWriter.write_field` calls.
        """
        ...

    fn write_header(read self, mut writer: DelimWriter) raises:
        """Write `self`s headers to the passed in `writer`.

        This should probably be done with `DelimWriter.write_record` or a series of
        `DelimWriter.write_field` calls.
        """
        ...


struct DelimWriter[W: Movable & Writer](Movable):
    """Write delimited data."""

    var delim: String
    """The delimiter to use."""
    var writer: BufferedWriter[Self.W]
    """The `BufferedWriter` to write to."""
    var write_header: Bool
    """Whether or not to write headers."""
    var needs_to_write_header: Bool
    """Whether or not we need to write the headers still."""

    fn __init__(
        out self,
        var writer: BufferedWriter[Self.W],
        *,
        var delim: String,
        write_header: Bool,
    ) raises:
        """Create a `DelimWriter`.

        Args:
            writer: The `BufferedWriter` to write to.
            delim: The delimiter to use.
            write_header: Whether or not to write headers.
        """
        self.writer = writer^
        self.delim = delim^
        self.write_header = write_header
        self.needs_to_write_header = write_header

    fn __moveinit__(out self, deinit existing: Self):
        self.delim = existing.delim^
        self.writer = existing.writer^
        self.write_header = existing.write_header
        self.needs_to_write_header = existing.needs_to_write_header

    fn __enter__(var self) -> Self:
        return self^

    fn flush(mut self):
        self.writer.flush()

    fn close(mut self) raises:
        self.flush()
        self.writer.close()

    fn write_record[*Ts: Writable](mut self, *args: *Ts) raises:
        """Write the passed in arguments as a delimited record."""

        @parameter
        fn write_elem[index: Int, T: Writable](arg: T):
            arg.write_to(self.writer)

            @parameter
            if index == args.__len__() - 1:
                self.writer.write("\n")
            else:
                self.writer.write(self.delim)

        @parameter
        for i in range(0, args.__len__()):
            write_elem[i](args[i])

    fn write_field[T: Writable](mut self, column: T, *, is_last: Bool) raises:
        """Write a single field, delimited by the configured delimiter."""
        column.write_to(self.writer)
        if not is_last:
            self.writer.write(self.delim)
        else:
            self.writer.write("\n")

    fn serialize[T: ToDelimited](mut self, read value: T) raises:
        """Write a struct that implements `ToDelimted` to the underlying writer.
        """
        if self.needs_to_write_header:
            value.write_header(self)
            self.needs_to_write_header = False
        value.write_to_delimited(self)
