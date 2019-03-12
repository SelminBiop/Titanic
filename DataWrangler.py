import numpy as np


class DataWrangler:

    def wrangle(self, dataframe):
        # Let's engineer two new features, FamilySize and if the passenger was alone or not (could influence how easy it was
        # to find a place on a boat for everyone)
        dataframe["FamilySize"] = dataframe["SibSp"] + dataframe["Parch"]
        # IsAlone is a bit more complicated since a child could be accompanied by a nanny or an aunt but still have a
        # FamilySize of 0, we'll assume that everyone under the age of 18 was accompanied
        dataframe["IsAlone"] = dataframe["FamilySize"].apply(lambda fam_size: 0 if fam_size > 0 else 1)
        is_old_enough = dataframe["Age"].apply(lambda age: 0 if age < 18 else 1)
        dataframe["IsAlone"] = dataframe["IsAlone"] * is_old_enough

        # Taken from https://medium.com/i-like-big-data-and-i-cannot-lie/how-i-scored-in-the-top-9-of-kaggles-titanic-machine-learning-challenge-243b5f45c8e9
        # but I don't really like it. Too many assumptions. Same with normalized titles, what if a title is not in our map...
        dataframe["Title"] = dataframe["Name"].apply(
            lambda name: name.split(',')[1].split('.')[0].strip())
        normalized_titles = {
            "Capt": "Officer",
            "Col": "Officer",
            "Major": "Officer",
            "Jonkheer": "Royalty",
            "Don": "Royalty",
            "Sir": "Royalty",
            "Dr": "Officer",
            "Rev": "Officer",
            "the Countess": "Royalty",
            "Dona": "Royalty",
            "Mme": "Mrs",
            "Mlle": "Miss",
            "Ms": "Mrs",
            "Mr": "Mr",
            "Mrs": "Mrs",
            "Miss": "Miss",
            "Master": "Master",
            "Lady": "Royalty"
        }
        dataframe["Title"] = dataframe["Title"].map(normalized_titles)
        grouped = dataframe.groupby(["Sex", "Pclass", "Title"])

        dataframe["Age"] = grouped["Age"].apply(lambda x: x.fillna(x.median()))

        # Also, let's transform them into age categories. It makes more sense that way
        # ToDo : After careful data analysis try to find best age categories
        dataframe["Age"] = dataframe["Age"].astype(np.int64)
        dataframe["Age"] = dataframe["Age"].apply(
            lambda age:
            1 if age < 5 else
            2 if 5 <= age < 10 else
            3 if 10 <= age < 18 else
            4 if 18 <= age < 25 else
            5 if 25 <= age < 35 else
            6 if 35 <= age < 45 else
            7
        )

        dataframe["Fare"].fillna(dataframe["Fare"].dropna().median(), inplace=True)
        dataframe["Fare"] = dataframe["Fare"].apply(
            lambda fare:
            1 if fare <= 8 else
            2 if 8 < fare <= 14 else
            3 if 14 < fare <= 31 else
            4
        )
        dataframe["Fare"] = dataframe["Fare"].astype(np.int64)

        # Some Embarked entries are missing, let's try and replace them with where most of the people embarked from
        # ToDo : Would it be possible to do something similar to Age?
        dataframe["Embarked"].fillna(
            dataframe["Embarked"].value_counts().index[0],
            inplace=True
        )

        # The cabin (first letter) can be indicative of where someone was situated, close to a life boat maybe?
        # We fill with U for unknown
        dataframe["Cabin"].fillna("U", inplace=True)
        dataframe["Cabin"] = dataframe["Cabin"].apply(lambda cabin: cabin[0])

        # Transformation for the benefit of Tensorflow
        dataframe["Pclass"].apply(np.int64)

        dataframe = dataframe.drop("SibSp", axis=1)
        dataframe = dataframe.drop("Parch", axis=1)
        dataframe = dataframe.drop("Name", axis=1)
        dataframe = dataframe.drop("Ticket", axis=1)
        dataframe = dataframe.drop("PassengerId", axis=1)

        return dataframe


