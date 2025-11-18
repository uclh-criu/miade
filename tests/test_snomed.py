# Module for tests related to SNOMED relationships

import pandas

from miade.snomed import Transitive


def test_transitive():
    df = pandas.DataFrame(
        {
            "ancestorId": ["heart disease", "heart disease", "infection", "infection", "lung disease", "lung disease"],
            "descendantId": ["angina", "endocarditis", "endocarditis", "pneumonia", "pneumonia", "asthma"],
        }
    )

    transitive = Transitive(df)
    assert transitive.get_ancestorIds(set(["angina"])) == {"heart disease"}
    assert transitive.get_ancestorIds(set(["angina", "endocarditis"])) == {"heart disease", "infection"}
    assert transitive.get_descendantIds(set(["infection"])) == {"endocarditis", "pneumonia"}
    assert transitive.get_descendantIds(set(["infection", "lung disease"])) == {"endocarditis", "pneumonia", "asthma"}


def test_transitive_creation_from_int():
    df = pandas.DataFrame({"ancestorId": [1, 1, 2, 2, 3, 3], "descendantId": [4, 5, 5, 6, 6, 7]})

    transitive = Transitive(df)
    assert transitive.get_ancestorIds(set(["4"])) == {"1"}
    assert transitive.get_ancestorIds(set(["4", "5"])) == {"1", "2"}
    assert transitive.get_descendantIds(set(["2"])) == {"5", "6"}
    assert transitive.get_descendantIds(set(["2", "3"])) == {"5", "6", "7"}
