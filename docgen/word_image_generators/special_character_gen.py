from docgen.datasets.image_folder import NaiveImageFolder

class SpecialCharacterGenerator(NaiveImageFolder):

    def __init__(self, *args, character, **kwargs):
        super().__init__(*args, **kwargs)
        self.character = character

    def get(self, index=None):
        return {"img": super().get(index),
                "text": self.character
                }