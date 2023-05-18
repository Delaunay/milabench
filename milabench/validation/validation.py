from contextlib import contextmanager
from dataclasses import dataclass, field


class ValidationLayer:
    """Validation layer interface, captures events, makes report"""

    def __init__(self) -> None:
        self.early_stop = False
        
    def __call__(self, entry):
        return self._on_event(entry)

    def _on_event(self, data):
        self.early_stop = False

        data = dict(data)
        run = data.pop("#run", None)
        pack = data.pop("#pack", None)
        tg = ".".join(run["tag"]) if run else pack.config["name"]
        ks = set(data.keys())
        
        if data.event == "stop":
            self.early_stop = True
            return

        self.on_event(pack, run, tg, ks, data)

    def on_event(self, pack, run, tag, keys, data):
        raise NotImplementedError()

    def report(self, summary, **kwargs):
        raise NotImplementedError()

    def __enter__(self):
        return self
    
    def __exit__(self, *args, **kwargs):
        pass


class Summary:
    """Simple utility to generate report with subsections"""

    @dataclass
    class _Section:
        name: str
        body: list = field(default_factory=list)

        @property
        def is_empty(self):
            return len(self.body) == 0

    def __init__(self) -> None:
        self.root = Summary._Section("")
        self.stack = [self.root]
        self.sections_lookup = dict()
        self.indent = "  "
        self.has_content = False
        self.sections = ["=", "-", "^", "*", "`"]

    def _line_char(self, depth):
        return self.sections[min(depth, len(self.sections))]

    def is_empty(self):
        return not self.has_content

    @contextmanager
    def section(self, title):
        self.newsection(title)
        yield
        self.endsection()

    def newsection(self, title):
        s = self.sections_lookup.get(title)
        if s is None:
            s = Summary._Section(title)
            self.stack[-1].body.append(s)

        self.stack.append(s)
        self.sections_lookup[title] = s

    def endsection(self):
        self.stack.pop()

    def newline(self):
        self.stack[-1].body.append("")

    def underline(self, size, char=None, depth=None):
        if char is None:
            char = self._line_char(depth)
        return char * size

    def add(self, txt):
        self.has_content = True
        self.stack[-1].body.append(txt)

    def show(self, printfun=print):

        if self.has_content:
            output = []
            self._show(self.root.body, 0, output)
            output.append("")
            printfun("\n".join(output))

    def _show(self, body, depth, output):
        def newline(text):
            output.append(self.indent * depth + text)

        for line in body:
            if isinstance(line, Summary._Section):
                if line.is_empty:
                    continue

                newline(line.name)
                newline(self.underline(len(line.name), depth=depth))
                self._show(line.body, depth + 1, output)
                continue

            newline(line)