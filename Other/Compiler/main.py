"""

https://blog.usejournal.com/writing-your-own-programming-language-and-compiler-with-python-a468970ae6df

Useful links
	https://www.aosabook.org/en/500L/a-python-interpreter-written-in-python.html
	https://tomassetti.me/ebnf/
	https://en.wikipedia.org/wiki/Extended_Backus%E2%80%93Naur_form
	https://en.wikipedia.org/wiki/Abstract_syntax_tree
	https://llvmlite.readthedocs.io/en/latest/
	https://buildmedia.readthedocs.org/media/pdf/rply/latest/rply.pdf
	https://www.pythonmembers.club/2018/05/01/building-a-lexer-in-python-tutorial/
	https://tomassetti.me/guide-parsing-algorithms-terminology/
"""
from lexer import Lexer

text_input = """
print(4 + 4 - 2);
"""

lexer = Lexer().get_lexer()
tokens = lexer.lex(text_input)

for token in tokens:
    print(token)
view raw

