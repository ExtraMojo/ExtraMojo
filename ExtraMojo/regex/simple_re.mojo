"""
A very simple regex implementation in Mojo inspired by Rob Pikes implementation.

## References
- https://www.cs.princeton.edu/courses/archive/spr09/cos333/beautiful.html
"""
from builtin.dtype import DType

alias START_ANCHOR = ord("^")
alias END_ANCHOR = ord("$")
alias DOT = ord(".")
alias STAR = ord("*")
alias NULL = 0


fn is_match(read regexp: String, read text: String) -> Bool:
    """
    Search for regexp anywhere in text and return true if it matches.

    ```mojo
    from testing import assert_true, assert_false
    from ExtraMojo.regex.simple_re import is_match

    var re = "wha*"
    assert_true(is_match(re, "what am I doing here"))
    assert_true(is_match(re, "whaaaaaaat am I doing here"))
    assert_true(is_match(re, "wht am I doing here"))
    assert_false(is_match(re, "wt am I doing here"))
    ```

    Args:
        regexp: The regular expression to search with.
        text: The text to search.

    Returns:
        True if the `regexp` matches the `text`.
    """
    var re = regexp.as_bytes()
    var txt = text.as_bytes()

    return is_match_bytes(re, txt)


fn is_match_bytes(regexp: Span[UInt8], text: Span[UInt8]) -> Bool:
    """
    Search for regexp anywhere in text and return true if it matches.

    ```mojo
    from testing import assert_true, assert_false
    from ExtraMojo.regex.simple_re import is_match_bytes

    var re = "^cat".as_bytes()
    assert_true(is_match_bytes(re, "cats of a feather".as_bytes()))
    assert_false(is_match_bytes(re, "bird cats of a cat".as_bytes()))
    ```

    Args:
        regexp: The regular expression to search with.
        text: The text to search.

    Returns:
        True if the `regexp` matches the `text`.
    """
    var txt = text
    var re = regexp
    if re[0] == START_ANCHOR:
        return _is_match_here(re[1:], txt)

    while True:
        # Must look even if string is empty
        if _is_match_here(re, txt):
            return True
        if txt[0] == NULL:
            break
        txt = txt[1:]

    return False


fn _is_match_here(regexp: Span[UInt8], text: Span[UInt8]) -> Bool:
    """
    Search for regexp at beginning of text.
    """
    if regexp[0] == NULL:
        return True
    if regexp[1] == STAR:
        return _is_match_star(regexp[0], regexp[2:], text)
    if regexp[0] == END_ANCHOR and regexp[1] == NULL:
        return text[0] == NULL
    if text[0] != NULL and (regexp[0] == DOT or regexp[0] == text[0]):
        return _is_match_here(regexp[1:], text[1:])
    return False


fn _is_match_star(c: UInt8, regexp: Span[UInt8], text: Span[UInt8]) -> Bool:
    """
    Search for c*regexp at beginning of text.
    """
    var txt = text
    while True:
        # a `*` matches zero or more instances
        if _is_match_here(regexp, txt):
            return True

        if txt[0] == NULL or (txt[0] != c and c != DOT):
            break
        txt = txt[1:]
    return False
