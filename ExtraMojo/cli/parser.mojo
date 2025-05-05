"""A very basic CLI Opt Parser.

```mojo {doctest="parser" class="no-wrap"}
from testing import assert_equal, assert_true
from ExtraMojo.cli.parser import OptParser, OptConfig, OptKind

var args = List(String("--file"), String("/path/to/thing"), String("--count"), String("42"), String("--fraction"), String("-0.2"), String("--verbose"), String("ExtraFile.tsv"))
var program_name = "example"

var parser = OptParser(name="example", description="An example program.")
parser.add_opt(OptConfig("file", OptKind.StringLike, default_value=None, description="A file with something in it."))
parser.add_opt(OptConfig("count", OptKind.IntLike, default_value=String("100"), description="A number."))
parser.add_opt(OptConfig("fraction", OptKind.FloatLike, default_value=String("0.5"), description="Some interesting fraction to keep."))
# Note that with flags, the OptKind must be BoolLike and there must be a default_value specified.
parser.add_opt(OptConfig("verbose", OptKind.BoolLike, is_flag=True, default_value=String("False"), description="Turn up the logging."))
# Specify any needed arguments. If at least n arguments aren't found after parsing the opts, an exception will be raised.
parser.expect_at_least_n_args(1, "Additional files to process")

# Note, a user would call parser.parse_sys_args()
var opts = parser.parse_args(args)

assert_equal(opts.get_string("file"), String("/path/to/thing"))
assert_equal(opts.get_int("count"), 42)
assert_equal(opts.get_float("fraction"), -0.2)
assert_equal(opts.get_bool("verbose"), True)
assert_true(len(opts.get_help_message()[]) > 0)
```
"""
from collections import Dict

import sys


@value
struct OptKind:
    """The viable types for an option to have."""

    var value: UInt8

    alias StringLike = Self(0)
    alias IntLike = Self(1)
    alias FloatLike = Self(2)
    alias BoolLike = Self(3)

    fn __init__(out self, value: UInt8):
        self.value = value

    fn __eq__(self, read other: Self) -> Bool:
        return self.value == other.value

    fn __ne__(self, read other: Self) -> Bool:
        return not self == other

    fn __str__(read self) -> String:
        if self == Self.BoolLike:
            return "Bool"
        elif self == Self.IntLike:
            return "Int"
        elif self == Self.FloatLike:
            return "Float"
        else:
            # elif self == Self.StringLike:
            return "String"


@value
struct OptValue:
    """When an option is parsed, it's stored as an OptValue.

    To get concrete values out of the `ParsedOpts` prefer to use the
    `ParsedOpts.get_<type>()` methods.
    """

    var kind: OptKind
    var _string: Optional[String]
    var _int: Optional[Int]
    var _float: Optional[Float64]
    var _bool: Optional[Bool]

    fn __init__(out self, str_value: String = ""):
        self.kind = OptKind.StringLike
        self._string = str_value
        self._int = None
        self._float = None
        self._bool = None

    fn __init__(out self, int_value: Int = 0):
        self.kind = OptKind.IntLike
        self._string = None
        self._int = int_value
        self._float = None
        self._bool = None

    fn __init__(out self, float_value: Float64 = 0.0):
        self.kind = OptKind.FloatLike
        self._string = None
        self._int = None
        self._float = float_value
        self._bool = None

    fn __init__(out self, bool_value: Bool = False):
        self.kind = OptKind.BoolLike
        self._string = None
        self._int = None
        self._float = None
        self._bool = bool_value

    # TODO: there's currently no good way to make a more generic compile time
    # `get[ReturnType: AnyType](read self) -> ReturnType` that can use the `ReturnType`
    # with `@parameter` to select what to return since types aren't comparable.
    fn get_string(read self) -> Optional[String]:
        return self._string

    fn get_int(read self) -> Optional[Int]:
        return self._int

    fn get_float(read self) -> Optional[Float64]:
        return self._float

    fn get_bool(read self) -> Optional[Bool]:
        return self._bool

    @staticmethod
    fn parse_string(read value: String) -> Self:
        return Self(value)

    @staticmethod
    fn parse_int(read value: String) raises -> Self:
        return Self(atol(value))

    @staticmethod
    fn parse_float(read value: String) raises -> Self:
        return Self(atof(value))

    @staticmethod
    fn parse_bool(read value: String) raises -> Self:
        if value.lower() == "true":
            return Self(True)
        elif value.lower() == "false":
            return Self(False)
        elif value.lower() == "1":
            return Self(True)
        elif value.lower() == "0":
            return Self(False)

        raise String("Attempt to covert invalid bool value of {}").format(value)

    @staticmethod
    fn parse_kind(kind: OptKind, read value: String) raises -> Self:
        """Parse the string based on the value of `OptKind`."""
        if kind == OptKind.BoolLike:
            return OptValue.parse_bool(value)
        elif kind == OptKind.StringLike:
            return OptValue.parse_string(value)
        elif kind == OptKind.IntLike:
            return OptValue.parse_int(value)
        elif kind == OptKind.FloatLike:
            return OptValue.parse_float(value)
        else:
            raise "Unsupported OptKind"


@value
struct OptConfig:
    """Create an option to be added to the `OptParser`."""

    var long_name: String
    """Required long name of, this will be used as the cli value as `--long_name`."""
    var default_value: Optional[String]
    """If there is one, the Stringified deafult value. This will be parsed via `OptKind`."""
    var value_kind: OptKind
    """The type of the value for this option."""
    var is_flag: Bool
    """If it's a flag, then it's value_kind needs to be Bool."""
    var description: String
    """Long for description, for best results, don't add a newline."""

    fn __init__(
        out self,
        owned long_name: String,
        value_kind: OptKind,
        *,
        is_flag: Bool = False,
        owned description: String = "",
        owned default_value: Optional[String] = None,
    ) raises:
        if is_flag and value_kind != OptKind.BoolLike and default_value:
            raise "If a DefinedArg `is_flag=True`, then the type of the argument must be `OptKind.BoolLike`, and a `default_value` must be supplied."
        self.long_name = long_name^
        self.value_kind = value_kind
        self.is_flag = is_flag
        self.description = description
        self.default_value = default_value


@value
struct ParsedOpts:
    """The parsed CLI options. Access your values with `ParsedOpts.get_string()`, `ParsedOpts.get_int()`, etc.

    Access CLI arguments from `ParsedOpts.args`.
    Get the help message with `ParsedOpts.get_help_message`.

    Note that there is an automatic `help` flag added to your options, it can be overridden by another option with that same name.
    The input args are first scanned for "--help" and if that is found the parser will exit early, returning the parsed value of the
    "--help" flag (or option if you have overridden it). It is up to the user to check for the "help" option being set and print the
    help message.
    """

    var options: Dict[String, OptValue]
    var args: List[String]
    var help_msg: String

    fn __init__(out self, owned help_msg: String = ""):
        self.options = Dict[String, OptValue]()
        self.args = List[String]()
        self.help_msg = help_msg

    fn get_help_message(
        ref self,
    ) -> Pointer[String, __origin_of(self.help_msg)]:
        """Get a nicely formatted help string."""
        return Pointer(to=self.help_msg)

    fn get_string(read self, read key: String) raises -> String:
        """Try to get the option specified with the given key as a String.

        This will raise if the key is not found, or if the type of the option doesn't match asked-for type.
        """
        var opt = self.options.get(key)
        if not key:
            raise String.write(key, " not found in options")

        var str_value = opt.value().get_string()
        if not str_value:
            raise String.write(
                "No string value for ",
                key,
                ". Check the specified option type.",
            )
        return str_value.value()

    fn get_int(read self, read key: String) raises -> Int:
        """Try to get the option specified with the given key as an Int.

        This will raise if the key is not found, or if the type of the option doesn't match asked-for type.
        """
        var opt = self.options.get(key)
        if not key:
            raise String.write(key, " not found in options")

        var int_value = opt.value().get_int()
        if not int_value:
            raise String.write(
                "No Int value for ", key, ". Check the specified option type."
            )
        return int_value.value()

    fn get_float(read self, read key: String) raises -> Float64:
        """Try to get the option specified with the given key as a Float64.

        This will raise if the key is not found, or if the type of the option doesn't match asked-for type.
        """
        var opt = self.options.get(key)
        if not key:
            raise String.write(key, " not found in options")

        var float_value = opt.value().get_float()
        if not float_value:
            raise String.write(
                "No Float64 value for ",
                key,
                ". Check the specified option type.",
            )
        return float_value.value()

    fn get_bool(read self, read key: String) raises -> Bool:
        """Try to get the option specified with the given key as a Bool.

        This will raise if the key is not found, or if the type of the option doesn't match asked-for type.
        """
        var opt = self.options.get(key)
        if not key:
            raise String.write(key, " not found in options")

        var bool_value = opt.value().get_bool()
        if not bool_value:
            raise String.write(
                "No Bool value for ", key, ". Check the specified option type."
            )
        return bool_value.value()


@value
struct OptParser:
    """[`OptParser`] will try to parse your long-form CLI options."""

    var options: Dict[String, OptConfig]
    """The options this will attempt to parse."""
    var program_description: String
    """The description of the program, to be used in the help message."""
    var program_name: String
    """Your programs name, to be used in the help message."""
    var min_num_args_expected: Optional[Int]
    """Whether or not to expect any program args, and if so, at least (>=) how many."""
    var args_help_msg: String
    """Help message for arguments."""

    fn __init__(
        out self,
        *,
        name: String,
        description: String = "",
    ) raises:
        self.options = Dict[String, OptConfig]()
        self.program_description = description
        self.program_name = name
        self.min_num_args_expected = None
        self.args_help_msg = ""

        # Add help message by default, this means a user can override help if they want to
        self.add_opt(
            OptConfig(
                "help",
                OptKind.BoolLike,
                is_flag=True,
                default_value=String("False"),
                description="Show help message",
            )
        )

    fn expect_at_least_n_args(mut self, n: Int, args_help_msg: String = ""):
        """The minimum number of args to expect.

        len(args) >= min_num_args_expected

        If not set, args are not checked.
        """
        self.min_num_args_expected = n
        self.args_help_msg = args_help_msg

    fn add_opt(mut self, owned arg: OptConfig):
        """Add an [`OptConfig`]."""
        self.options[arg.long_name] = arg

    fn help_msg(read self) -> String:
        """Get the help message string based on the currently added options."""

        @parameter
        fn write_arg_msg(mut writer: String, read opt: OptConfig):
            writer.write(
                "\t--",
                opt.long_name,
                " <",
                String(opt.value_kind),
                ">",
            )
            if opt.default_value:
                writer.write(" [Default: ", opt.default_value.value(), "]")
            else:
                writer.write(" [Required]")
            writer.write("\n")
            writer.write("\t\t", opt.description, "\n\n")

        var help_msg = String()
        help_msg.write(self.program_name, "\n")
        help_msg.write(self.program_description, "\n\n")

        if self.min_num_args_expected:
            help_msg.write("ARGS:\n")
            help_msg.write(
                "\t", "<ARGS (>=", self.min_num_args_expected.value(), ")>...\n"
            )
            if len(self.args_help_msg) > 0:
                help_msg.write("\t\t", self.args_help_msg, "\n")

        help_msg.write("FLAGS:\n")
        for kv in self.options.items():
            if not kv[].value.is_flag:
                continue
            write_arg_msg(help_msg, kv[].value)

        help_msg.write("OPTIONS:\n")
        for kv in self.options.items():
            if kv[].value.is_flag:
                continue
            write_arg_msg(help_msg, kv[].value)
        # TODO: create a USAGE message if we every add anything about expected ARGS
        return help_msg

    @staticmethod
    fn _strip_leading_dashes(
        arg: StringSlice,
    ) raises -> String:
        # TODO: use a string slice or something better here
        var i = 0
        while i < len(arg):
            if arg[i] != "-":
                break
            i += 1
        return String(arg[i:])

    fn parse_sys_args(mut self) raises -> ParsedOpts:
        """Parse the arguments from `sys.argv()`."""
        var args = sys.argv()

        var exe = args[0]
        if not self.program_name:
            self.program_name = exe.__str__()

        var fixed = List[String]()
        var i = 1  # Skip the first arg
        while i < len(args):
            fixed.append(args[i].__str__())
            i += 1
        return self.parse_args(fixed)

    fn parse_args(read self, args: List[String]) raises -> ParsedOpts:
        """Parse the arguments passed in via `args`."""
        var result = ParsedOpts(help_msg=self.help_msg())

        # Short circuit if "--help" is found
        var j = 0
        for arg in args:
            if arg[] == "--help":
                var opt = "help"
                # Even though help can be overridden, we treat it specially
                var opt_def = self.options.get("help")
                if opt_def:
                    var opt_def = opt_def.value()
                    if not opt_def.is_flag:
                        j += 1
                        if j >= len(args):
                            raise String.write(
                                "Missing value for option: ", opt
                            )
                        var value = args[j]
                        # Get the value from the next string
                        result.options[opt] = OptValue.parse_kind(
                            opt_def.value_kind, value
                        )

                    else:
                        # It's a flag! invert the default value
                        result.options[opt] = OptValue(
                            not OptValue.parse_bool(
                                opt_def.default_value.value()
                            )
                            .get_bool()
                            .value()
                        )
                return result
            j += 1

        var i = 0
        while i < len(args):
            var arg = args[i]
            if arg.startswith("-"):
                var opt = self._strip_leading_dashes(arg)
                var opt_def = self.options.get(opt)
                if opt_def:
                    var opt_def = opt_def.value()
                    if not opt_def.is_flag:
                        i += 1
                        if i >= len(args):
                            raise String.write(
                                "Missing value for option: ", opt
                            )
                        var value = args[i]
                        # Get the value from the next string
                        result.options[opt] = OptValue.parse_kind(
                            opt_def.value_kind, value
                        )

                    else:
                        # It's a flag! invert the default value
                        result.options[opt] = OptValue(
                            not OptValue.parse_bool(
                                opt_def.default_value.value()
                            )
                            .get_bool()
                            .value()
                        )
                else:
                    raise String.write("No such option: ", opt)
            else:
                result.args.append(arg)
            i += 1

        # Fill in any args that haven't been seen with their defaults
        for arg in self.options.items():
            if not result.options.get(arg[].key):
                var default = arg[].value.default_value
                if not default:
                    raise String.write("No value provided for ", arg[].key)
                result.options[arg[].key] = OptValue.parse_kind(
                    arg[].value.value_kind, default.value()
                )

        if self.min_num_args_expected:
            if len(result.args) < self.min_num_args_expected.value():
                raise String("Expected >= {} arguments, found {}").format(
                    self.min_num_args_expected.value(),
                    len(result.args),
                )

        return result


@value
struct Subcommand(Hashable):
    """A subcommand.

    The name of the subcommand is the `OptParser.name`.
    The name of the subcommand will be checked against the first value in the input args.
    The help message will display the program description for the `OptParser` that is associated with this subcommand.
    """

    var parser: OptParser

    fn __init__(out self, owned name: String, owned parser: OptParser):
        self.parser = parser^

    fn __hash__(read self) -> UInt:
        return self.parser.program_name.__hash__()

    fn __eq__(read self, read other: Self) -> Bool:
        return self.parser.program_name == other.parser.program_name

    fn __ne__(read self, read other: Self) -> Bool:
        return not (self == other)


@value
struct SubcommandParser:
    """Subcommands are created by passing in the command, and an `OptParser`.

    The parser is for the options for the subcommand.

    ```mojo
    from testing import assert_equal, assert_true
    from ExtraMojo.cli.parser import OptParser, OptConfig, OptKind, SubcommandParser, Subcommand

    var args = List(String("do-work"), String("--file"), String("/path/to/thing"), String("--count"), String("42"), String("--fraction"), String("-0.2"), String("--verbose"))
    var program_name = "example"

    var parser = OptParser(name="do-work", description="An example program.")
    parser.add_opt(OptConfig("file", OptKind.StringLike, default_value=None, description="A file with something in it."))
    parser.add_opt(OptConfig("count", OptKind.IntLike, default_value=String("100"), description="A number."))
    parser.add_opt(OptConfig("fraction", OptKind.FloatLike, default_value=String("0.5"), description="Some interesting fraction to keep."))
    # Note that with flags, the OptKind must be BoolLike and there must be a default_value specified.
    parser.add_opt(OptConfig("verbose", OptKind.BoolLike, is_flag=True, default_value=String("False"), description="Turn up the logging."))

    var cmd = Subcommand(parser) # uses the name from the passed in parser
    var cmd_parser = SubcommandParser(name=String("cool-program"), description="Do some cool stuff.")
    cmd_parser.add_command(cmd)

    # Note, a user would call parser.parse_sys_args()
    var cmd_and_opts = cmd_parser.parse_args(args)
    if not cmd_and_opts:
        print(cmd_parser.get_help_message())
    parsed_cmd, opts = cmd_and_opts.value()


    if parsed_cmd == cmd.parser.program_name:
        assert_equal(opts.get_string("file"), String("/path/to/thing"))
        assert_equal(opts.get_int("count"), 42)
        assert_equal(opts.get_float("fraction"), -0.2)
        assert_equal(opts.get_bool("verbose"), True)
        assert_true(len(opts.get_help_message()[]) > 0)
    ```
    """

    var commands: Dict[String, Subcommand]
    var description: String
    var name: String

    fn __init__(
        out self,
        *,
        owned name: String,
        owned description: String = "",
    ):
        self.name = name^
        self.description = description^
        self.commands = Dict[String, Subcommand]()

    fn get_help_message(read self) raises -> String:
        """Create the help message for the subcommands."""
        var help = String()
        help.write(String("{}\n").format(self.name))
        help.write(String("\t{}\n\n").format(self.description))
        help.write("SUBCOMMANDS:\n")
        for kv in self.commands.items():
            help.write(
                String("\t{}: {}\n").format(
                    kv[].key, kv[].value.parser.program_description
                )
            )

        return help

    fn add_command(mut self, command: Subcommand):
        """Add a subcommand."""
        self.commands[command.parser.program_name] = command

    fn parse_args(
        read self, args: List[String]
    ) raises -> Optional[Tuple[String, ParsedOpts]]:
        """Parse the input args, expecting a subcommand."""
        if len(args) == 0:
            return None

        if len(self.commands) == 0:
            raise "No subcommands configured"

        var first_arg = args[0]
        var cmd = self.commands.get(first_arg)
        if not cmd:
            return None

        return (
            cmd.value().parser.program_name,
            cmd.value().parser.parse_args(args[1:]),
        )

    fn parse_sys_args(read self) raises -> Optional[Tuple[String, ParsedOpts]]:
        """Parse the sys.argv() list."""
        var args = sys.argv()

        var fixed = List[String]()
        var i = 1  # Skip the first arg which is the exe
        while i < len(args):
            fixed.append(args[i].__str__())
            i += 1
        return self.parse_args(fixed)
