from milabench.cli import main
from milabench.validation.validation import Summary


def test_report(runs_folder, capsys, file_regression, config):
    folder = runs_folder / "rijubigo.2023-03-24_13:45:27.512446"
    try:
        main(["report", "--runs", str(folder), "--config", config("benchio")])
    except SystemExit as exc:
        assert not exc.code
        assert exc.code is None

    all = capsys.readouterr()
    stdout = all.out
    assert stdout != ""

    stdout = stdout.replace(str(folder), "XXX")
    file_regression.check(stdout)


def test_summary(file_regression):
    benchs = ["matmult", "matsub"]
    points = [
        "1. Errors stuff happened",
        "2. Errors stuff happened",
        "3. Errors stuff happened",
    ]
    report = Summary()

    with report.section("Errors"):
        for bench in benchs:
            with report.section(bench):
                for p in points:
                    report.add(p)

    output = ""

    def get_output(data):
        nonlocal output
        output = data

    report.show()
    report.show(get_output)
    file_regression.check(output)


def test_empty_summary():
    report = Summary()
    with report.section("Errors"):
        with report.section("Bench"):
            pass

    output = ""

    def get_output(data):
        nonlocal output
        output = data

    report.show(get_output)

    assert output.strip() == ""


def test_report_folder_does_average(runs_folder, capsys, config, file_regression):
    try:
        main(["report", "--runs", runs_folder, "--config", config("benchio")])
    except SystemExit as exc:
        assert not exc.code

    output = capsys.readouterr().out
    output = output.replace(str(runs_folder), "XXX")
    file_regression.check(output)


def test_compare(runs_folder, capsys, file_regression):
    try:
        main(["compare", runs_folder])
    except SystemExit as exc:
        assert not exc.code

    output = capsys.readouterr().out
    output = output.replace(str(runs_folder), "XXX")
    file_regression.check(output)


def test_summary_full(runs_folder):
    from milabench.common import _read_reports, make_report, make_summary

    run = runs_folder / "rijubigo.2023-03-24_13:45:27.512446"

    runs = [run]
    reports = _read_reports(*runs)
    summary = make_summary(reports)

    make_report(summary, None)



def _read_reports_zip(folder):
    from milabench.common import XPath, _parse_report
    import zipfile

    with zipfile.ZipFile(folder, "r") as archive:
        all_data = {}

        for member in archive.namelist():
            if member.endswith(".data"):
                all_data[str(member)] = _parse_report(member, open=archive.open)

    return all_data


def test_summary_full_zip(runs_folder):
    import os
    from milabench.common import make_report, make_summary

    scenario = os.path.join(runs_folder, "8GPUs.zip")

    reports = _read_reports_zip(scenario)
    summary = make_summary(reports)

    make_report(summary, None)
