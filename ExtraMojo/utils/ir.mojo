"""
Package version of [this](https://forum.modular.com/t/seeing-the-mlir-llvm-or-asm-code-generated-by-mojo/250/5?u=duck_tape) from @soraros.
Linked [gist](https://gist.github.com/soraros/44d56698cb20a6c5db3160f13ca81675)

## Usage

```mojo
from sys.intrinsics import assume

from ExtraMojo.utils.ir import dump_ir

fn main():
  dump_ir[f, "out1"](dir="./")
  dump_ir[g, "out2"](dir="/tmp")


@export  # use `export` so get cleaner names
fn f(x: Int) -> Int:
  assume(0 <= x < 100)
  return max(1, x * 2)

@export
fn g(x: Int) -> Int:
  assume(0 <= x < 100)
  return x * 2 + Int(x == 0)
```

"""
import compile
from builtin._location import __call_location
from os.path import dirname, join


@value
struct Setting:
    var kind: StaticString
    var ext: StaticString


fn dump_ir[
    Fn: AnyTrivialRegType, //, f: Fn, name: StaticString = "out"
](dir: String = "/tmp"):
    alias l = List(Setting("llvm", "ll"), Setting("asm", "s"))
    print(compile.get_linkage_name[f]())
    # dir = dirname(__call_location[inline_count=0]().file_name)

    @parameter
    for i in range(len(l)):
        alias s = l[i]
        ir = compile.compile_info[f, emission_kind = s.kind]()
        print("-", full_path := join(dir, name + "." + s.ext))
        try:
            with open(full_path, "w") as f:
                f.write(ir)
        except e:
            print("error:", e)
