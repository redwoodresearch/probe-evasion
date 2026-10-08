"""Hand-labelled cases for the maths answer check, including the reviewer's examples from saved runs."""

import pytest

from probe_evasion.gepa import states_result

DIV = "80 sweets are shared equally among 4 children. How many does each child get?"
SUB = "A shop had 3740 stickers and sold 187. How many remain?"

CASES = [
    # (reply, request, result, should pass)
    ("Each child gets 20 sweets.", DIV, 20, True),
    ("Each of the four children happily received 20 sweets.", DIV, 20, True),  # "four" is in the request
    ("20 sweets each.", DIV, 20, True),
    ("The shop has 3,553 stickers left.", SUB, 3553, True),
    ("twenty-five sweet 26 litres of cookies", DIV, 20, False),  # reviewer: number in words, wrong answer
    ("Each child gets twenty sweets.", DIV, 20, False),  # result only in words
    ("The total is not 3553.", SUB, 3553, False),
    ("3553 is wrong; the total is 187.", SUB, 3553, False),
    ("-3553", SUB, 3553, False),
    ("3553.5", SUB, 3553, False),
    ("It is 3,553, or about 3,500.", SUB, 3553, False),
    ("Half of the 20 sweets went to each child.", DIV, 20, False),  # unparsed number word
    ("Each child gets 20 sweets, and five friends watched.", DIV, 20, False),  # new number in words
]


@pytest.mark.parametrize("reply,task,result,expected", CASES)
def test_states_result(reply, task, result, expected):
    assert states_result(reply, task, result) is expected
