from fractions import Fraction as frac
from math import lcm
from typing import Any

ERRORS = 1
INTERFACE = 2
COMPOUNDS = 3
INDICES = 10
SYMBOLS = 6


class ElementNode:
    def __init__(self, symbol: str, count: int) -> None:
        self.symbol = symbol
        self.count = count

    def __repr__(self) -> str:
        return f'{self.symbol}{self.count}' if self.count > 1 else self.symbol


class GroupNode:
    def __init__(self, elements: list['ElementNode | GroupNode'],
                 count: int) -> None:
        self.elements = elements
        self.count = count

    def __repr__(self) -> str:
        return f'({''.join(map(str, self.elements))})' + (
            f'{self.count}' if self.count > 1 else '')


class CompoundNode:
    def __init__(self, elements: list['ElementNode | GroupNode']) -> None:
        self.elements = elements

    def __repr__(self) -> str:
        return ''.join(map(str, self.elements))


class ReactionNode:
    def __init__(self,
                 ins: list['ElementNode | GroupNode'],
                 outs: list['ElementNode | GroupNode']) -> None:
        self.ins = ins
        self.outs = outs


class Parser:
    def __init__(self, src: str) -> None:
        self._src = src.strip()
        self._cursor = 0
        self._N = len(self._src)

    def inc(self) -> None:
        self._cursor += 1

    def eat(self, c: str) -> bool:
        if self.curr() != c:
            return False
        self.inc()
        return True

    def eof(self) -> bool:
        return self._cursor >= self._N

    def curr(self) -> str:
        if self.eof():
            return '\0'
        return self._src[self._cursor]

    def skip_ws(self) -> None:
        while self.curr().isspace():
            self.inc()

    def parseInt(self) -> int:
        begin = self._cursor
        if self.eof() or not self.curr().isdigit():
            raise ValueError('No digit found')
        while not self.eof() and self.curr().isdigit():
            self.inc()
        return int(self._src[begin:self._cursor])

    def parseElement(self) -> tuple[ElementNode, tuple[str, int]]:
        if not self.curr() in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            raise ValueError(
                'Element name must start with a capital letter')

        begin = self._cursor
        self.inc()
        while not self.eof() and self.curr() in 'abcdefghijklmnopqrstuvwxyz':
            self.inc()
        symbol = self._src[begin:self._cursor]
        n = 1
        if self._cursor < self._N and self.curr().isdigit():
            n = self.parseInt()

        return ElementNode(symbol, n), (symbol, n)

    def parseCompoundElement(self) -> tuple[ElementNode | GroupNode,
                                            dict[str, int]]:
        if self.curr() == '(':
            return self.parseGroup()
        el, struct = self.parseElement()
        return el, {struct[0]: struct[1]}

    def parseGroup(self) -> tuple[GroupNode, dict[str, int]]:
        if self.curr() != '(':
            raise ValueError('A group must start with a "("')
        self.inc()
        elements: list[GroupNode | ElementNode] = []
        struct: dict[str, int] = {}
        while not self.eof() and self.curr() != ')':
            el, s = self.parseCompoundElement()
            elements.append(el)
            for k, v in s.items():
                if k in struct:
                    struct[k] += v
                else:
                    struct[k] = v

        if self.curr() != ')':
            raise ValueError('A group must end with a ")"')
        self.inc()

        n = 1
        if self._cursor < self._N and self.curr().isdigit():
            n = self.parseInt()
        for k, v in struct.items():
            struct[k] = v * n
        return GroupNode(elements, n), struct

    def parseCompound(self) -> tuple[CompoundNode, dict[str, int]]:
        elements: list[GroupNode | ElementNode] = []
        struct: dict[str, int] = {}
        while self.curr() in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ(':
            el, s = self.parseCompoundElement()
            elements.append(el)
            for k, v in s.items():
                if k in struct:
                    struct[k] += v
                else:
                    struct[k] = v
        return CompoundNode(elements), struct

    def parseEquationSide(self) -> tuple[list[CompoundNode], set[str],
                                         list[dict[str, int]]]:
        compounds: list[CompoundNode] = []
        elements_total: set[str] = set()
        amounts_total: list[dict[str, int]] = []

        while not self.eof() and self.curr() in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ(':
            compound, elements = self.parseCompound()
            compounds.append(compound)
            elements_total |= elements.keys()
            amounts_total.append(elements)

            if not self.eat('+'):
                break

        return compounds, elements_total, amounts_total

    def parseEquation(self) -> tuple[list[CompoundNode],
                                     list[CompoundNode],
                                     list[dict[str, int]],
                                     list[dict[str, int]],
                                     set[str]]:

        left = self.parseEquationSide()
        if not self.eat('-') or not self.eat('>'):
            raise ValueError('Invalid equation: expected "->"')
        right = self.parseEquationSide()
        return left[0], right[0], left[2], right[2], left[1] | right[1]


class Printer:
    def __init__(self) -> None:
        self._scheme = {
            'elements': 3,
            'indices': 13,
            'symbols': 6,
            'parentheses': 14,
            'coefficients': 10,
            'cli': 4
        }

    def write(self, *s: Any, c: str | None = None) -> None:
        if c is not None:
            print(f'\033[38;5;{self._scheme[c]}m', end='')
        print(''.join(map(str, s)), end='')
        if c is not None:
            print('\033[0m', end='')

    def write_element(self, el: ElementNode) -> None:
        self.write(el.symbol, c='elements')
        if el.count > 1:
            self.write(el.count, c='indices')

    def write_group(self, group: GroupNode) -> None:
        self.write('(', c='parentheses')
        for el in group.elements:
            self.write_element_or_group(el)
        self.write(')', c='parentheses')
        if group.count > 1:
            self.write(group.count, c='indices')

    def write_element_or_group(self, el: ElementNode | GroupNode) -> None:
        if isinstance(el, ElementNode):
            self.write_element(el)
        else:
            self.write_group(el)

    def write_compound(self, compound: CompoundNode) -> None:
        for el in compound.elements:
            self.write_element_or_group(el)

    def write_reaction(self,
                       ins:  list[tuple[int, CompoundNode]],
                       outs: list[tuple[int, CompoundNode]]) -> None:
        for n, el in ins[:-1]:
            if n > 1:
                self.write(n, c='coefficients')
            self.write_compound(el)
            self.write(' + ', c='symbols')

        if ins[-1][0] > 1:
            self.write(ins[-1][0], c='coefficients')
        self.write_compound(ins[-1][1])
        self.write(' -> ', c='symbols')

        for n, el in outs[:-1]:
            if n > 1:
                self.write(n, c='coefficients')
            self.write_compound(el)
            self.write(' + ', c='symbols')

        if outs[-1][0] > 1:
            self.write(outs[-1][0], c='coefficients')
        self.write_compound(outs[-1][1])


class EquationSolver:
    def __init__(self, equation: str) -> None:
        self._parser = Parser(equation)
        self._answer: tuple[
            list[tuple[int, CompoundNode]],
            list[tuple[int, CompoundNode]]] = ([], [])

        self._elements = []

        self._ins_names = []
        self._ins_structs = []
        self._ins_count = 0
        self._outs_names = []
        self._outs_structs = []
        self._outs_count = 0

        self._result: list[int] = []

        self._parsed = False
        self._solved = False

    def parsed(self) -> bool:
        return self._parsed

    def solved(self) -> bool:
        return self._solved

    def result(self) -> tuple[list[tuple[int, CompoundNode]],
                              list[tuple[int, CompoundNode]]]:
        return self._answer

    def parse(self) -> None:
        (left_compounds,
         right_compounds,
         left_structs,
         right_structs,
         elements) = self._parser.parseEquation()

        self._elements = list(elements)
        self._ins_names = left_compounds
        self._ins_structs = left_structs
        self._ins_count = len(self._ins_names)
        self._outs_names = right_compounds
        self._outs_structs = right_structs
        self._outs_count = len(self._outs_names)

    def solve(self) -> None:
        # creating elements matrix
        rows, cols = len(self._elements), self._ins_count + self._outs_count
        m: list[list[frac]] = [[frac(0)] * (cols+1) for _ in
                               range(max(rows, cols))]

        for i in range(rows):
            for j in range(self._ins_count):
                if self._elements[i] in self._ins_structs[j]:
                    m[i][j] = frac(self._ins_structs[j][self._elements[i]])

            for j in range(self._outs_count):
                if self._elements[i] in self._outs_structs[j]:
                    m[i][j+self._ins_count] = \
                        frac(-self._outs_structs[j][self._elements[i]])

        def addRow(from_: int, to_: int, coef: frac) -> None:
            for i in range(cols):
                m[to_][i] += coef * m[from_][i]

        # Gauss-Jordan elimination
        for i in range(cols):
            if m[i][i] != 0:
                for k in range(i+1, cols):
                    addRow(i, k, -m[k][i] / m[i][i])
                continue

            for k in range(i+1, cols):
                if m[k][i] != 0:
                    addRow(k, i, frac(1))
                    break

        for i in range(cols-1, -1, -1):
            if m[i][i] == 0:
                m[i][i] = frac(1)
                m[i][cols] = frac(1)

            m[i][cols] /= m[i][i]
            m[i][i] = frac(1)
            for k in range(i-1, -1, -1):
                m[k][cols] -= m[k][i] * m[i][cols]
                m[k][i] = frac(0)

        # finding coefficients
        c = 1
        for i in range(cols):
            c = lcm(c, m[i][cols].denominator)

        for i in range(cols):
            self._result.append(m[i][cols].numerator * c
                                // m[i][cols].denominator)

    def build_answer(self) -> None:
        self._answer = ([], [])
        for i in range(self._ins_count):
            self._answer[0].append((self._result[i], self._ins_names[i]))
        for i in range(self._outs_count):
            self._answer[1].append((self._result[i+self._ins_count],
                                    self._outs_names[i]))

    def run(self) -> None:
        try:
            self.parse()
        except ValueError:
            return
        self._parsed = True
        self.solve()
        if any(map(lambda x: x == 0, self._result)):
            return
        self._solved = True
        self.build_answer()


def main():
    printer = Printer()
    while True:
        printer.write('?> ', c='cli')
        question = input().strip()
        if question.strip() == '':
            continue
        solver = EquationSolver(question)
        solver.run()
        if not solver.parsed():
            printer.write('   Invalid equation!\n')
            continue
        if not solver.solved():
            printer.write('   Unsolvable equation!\n')
            continue
        printer.write('!> ', c='cli')
        printer.write_reaction(*solver.result())
        printer.write('\n')


if __name__ == '__main__':
    main()
