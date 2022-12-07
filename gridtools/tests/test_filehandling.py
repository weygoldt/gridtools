import os

from gridtools.toolbox import filehandling

datapath = "tests/testdata"


def test_ConfLoader() -> None:

    conf = filehandling.ConfLoader(f"{datapath}/confloader_test.yml")

    assert isinstance(conf.var1, bool) and conf.var1 == True 
    assert isinstance(conf.var2, list) and conf.var2[0] == "string1"
    assert isinstance(conf.var3, float) and conf.var3 == 1.5
    assert isinstance(conf.var4, dict) and conf.var4["subvar3"] == 5
    assert isinstance(conf.var4["subvar3"], int)


def test_ListRecordings() -> None:

    path=f"{datapath}/mockgrid"
    exclude=["2022-04-20-18_49"]
    
    recs = filehandling.ListRecordings(path=path, exclude=exclude)

    assert recs.dataroot == path and isinstance(recs.dataroot, str)
    assert len(recs.recordings) == 1 and isinstance(recs.recordings, list)
    assert recs.recordings[0] == "2022-05-20-20_18"


def test_makeOutputdir() -> None:

    path = filehandling.makeOutputdir(path=f"{datapath}/testdir")
    assert os.path.isdir(path)
    os.rmdir(path)
