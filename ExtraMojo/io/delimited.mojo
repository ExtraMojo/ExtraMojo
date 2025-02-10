"""Working with simple delimited text.

## Example

TODO: update the examples to doctests when tempfile is done.

Compile-time known fields

```
from ExtraMojo.io.buffered import (
    BufferedReader,
    BufferedWriter,
)
from ExtraMojo.io.delimited import (
    DelimReader,
    FromDelimited,
    ToDelimited,
    DelimWriter,
)

@value
struct SerDerStruct(ToDelimited, FromDelimited):
    var index: Int
    var name: String

    fn write_to_delimited(read self, mut writer: DelimWriter) raises:
        writer.write_record(self.index, self.name)

    fn write_header(read self, mut writer: DelimWriter) raises:
        writer.write_record("index", "name")

    @staticmethod
    fn from_delimited(mut data: SplitIterator) raises -> Self:
        var index = Int(StringSlice(unsafe_from_utf8=data.__next__()))
        var name = String()  # String constructor expected nul terminated byte span
        name.write_bytes(data.__next__())
        return Self(index, name)


fn test_delim_reader_writer(file: Path) raises:
    var to_write = List[SerDerStruct]()
    for i in range(0, 1000):
        to_write.append(SerDerStruct(i, String("MyNameIs" + String(i))))
    var writer = DelimWriter(
        BufferedWriter(open(String(file), "w")), delim="\t", write_header=True
    )
    for item in to_write:
        writer.serialize(item[])
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
    print("Successful delim_writer")
```

Dynamic fields

```
@value
struct Score[
    is_mutable: Bool, //,
    truth_lengths_origin: Origin[is_mutable],
    truth_names_origin: Origin[is_mutable],
](ToDelimited):
    var assembly_name: String
    var assembly_length: Int
    var scores: List[Int32]
    var truth_lengths: Pointer[List[Int], truth_lengths_origin]
    var truth_names: Pointer[List[String], truth_names_origin]

    fn __init__(
        out self,
        owned assembly_name: String,
        assembly_length: Int,
        owned scores: List[Int32],
        ref [truth_lengths_origin]truth_lengths: List[Int],
        ref [truth_names_origin]truth_names: List[String],
    ):
        self.assembly_name = assembly_name^
        self.assembly_length = assembly_length
        self.scores = scores^
        self.truth_lengths = Pointer.address_of(truth_lengths)
        self.truth_names = Pointer.address_of(truth_names)

    fn write_to_delimited(read self, mut writer: DelimWriter) raises:
        writer.write_field(self.assembly_name, is_last=False)
        writer.write_field(self.assembly_length, is_last=False)
        for i in range(0, len(self.scores)):
            writer.write_field(
                "{}/{}".format(self.scores[i], self.truth_lengths[][i]),
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
    var assembly_fasta_file = opts.get_string("assembly-fasta-file")
    var truth_fasta_file = opts.get_string("truth-fasta-file")
    var output_scores_tsv = opts.get_string("output-scores-tsv")

    var assemblies = FastaRecord.slurp_fasta(assembly_fasta_file)
    var truths = FastaRecord.slurp_fasta(truth_fasta_file)

    var truth_lengths = List[Int](capacity=len(truths))
    var truth_names = List[String](capacity=len(truths))
    for truth in truths:
        truth_lengths.append(len(truth[].seq))
        truth_names.append(truth[].name)

    var out_writer = DelimWriter(
        BufferedWriter(open(output_scores_tsv, "w")),
        delim="\t",
        write_header=True,
    )

    for assembly in assemblies:
        var scores = List[Int32](capacity=len(truths))
        for truth in truths:
            var score = needleman_wunsch(
                assembly[].seq.as_bytes(), truth[].seq.as_bytes()
            )
            scores.append(score)
        var s = Score(
            assembly[].name,
            len(assembly[].seq),
            scores,
            truth_lengths,
            truth_names,
        )
        out_writer.serialize[
            Score[__origin_of(truth_lengths), __origin_of(truth_names)]
        ](s)

    out_writer.flush()
    out_writer.close()
```

"""
from collections import Optional
from memory import Span
from utils import Writer

from ExtraMojo.bstr.bstr import SplitIterator
from ExtraMojo.io.buffered import BufferedReader, BufferedWriter


trait FromDelimited(CollectionElement):
    """Create an instance of `Self` from the iterator over `Span[UInt8]` bytes.
    """

    @staticmethod
    fn from_delimited(mut data: SplitIterator) raises -> Self:
        ...


struct DelimReader[RowType: FromDelimited]:
    """Read delimited data that is delimited by a single bytes.

    The `RowType` must implement `FromBytes` which is passed an iterator over the split up line.
    """

    # TODO: there's something a bit odd about how this all works as an iterator.
    # I can't add a context manager, and the iterator has to use `^` which I'm not sure about.

    var delim: UInt8
    var reader: BufferedReader
    var next_elem: Optional[RowType]
    var buffer: List[UInt8]
    var len: Int
    var has_header: Bool

    fn __init__(
        out self,
        owned reader: BufferedReader,
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
        if self.has_header:
            self._skip_header()
        self._get_next()

    fn __moveinit__(out self, owned existing: Self):
        self.delim = existing.delim
        self.reader = existing.reader^
        self.next_elem = existing.next_elem^
        self.buffer = existing.buffer^
        self.len = existing.len
        self.has_header = existing.has_header

    fn __len__(read self) -> Int:
        return self.len

    fn __has_next__(read self) -> Bool:
        return self.__len__() > 0

    fn __next__(mut self) raises -> RowType:
        if not self.next_elem:
            raise "Attempting to call past end of iterator"
        var ret = self.next_elem.take()
        self._get_next()
        return ret

    fn __iter__(owned self) -> Self:
        return self^

    fn _skip_header(mut self) raises:
        var bytes_read = self.reader.read_until(self.buffer, ord("\n"))
        if bytes_read == 0:
            self.len = 0
            self.next_elem = None
            raise "No header found"

    fn _get_next(mut self) raises:
        var bytes_read = self.reader.read_until(self.buffer, ord("\n"))
        var tmp = String()
        tmp.write_bytes(self.buffer)
        if bytes_read == 0:
            self.len = 0
            self.next_elem = None
            return
        var iterator = SplitIterator(self.buffer, self.delim)
        self.next_elem = RowType.from_delimited(iterator)


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


struct DelimWriter:
    """Write delimited data."""

    var delim: String
    """The delimiter to use."""
    var writer: BufferedWriter
    """The `BufferedWriter` to write to."""
    var write_header: Bool
    """Whether or not to write headers."""
    var needs_to_write_header: Bool
    """Whether or not we need to write the headers still."""

    fn __init__(
        out self,
        owned writer: BufferedWriter,
        *,
        owned delim: String,
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

    fn __moveinit__(out self, owned existing: Self):
        self.delim = existing.delim^
        self.writer = existing.writer^
        self.write_header = existing.write_header
        self.needs_to_write_header = existing.needs_to_write_header

    fn __enter__(owned self) -> Self:
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

        args.each_idx[write_elem]()

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
