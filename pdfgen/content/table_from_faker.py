from faker.ancestry.custom_row import RowBuilder
import random

class TableDataFromFaker:
    """ Generate content
    """
    def __init__(self, functions,
                 header_names=None,
                 include_row_number=True,
                 provider_dict_list=None,
                 mark_for_replacement_fields=None,
                 replacement_char=None,
                 random_fields=0,
                 ):
        """ Random fields will be sampled WITH REPLACEMENT

        """
        self.include_row_number = include_row_number
        if header_names is None:
            self.header_names = self.clean_fnc_name(functions)
            if include_row_number:
                self.header_names.insert(0, "Row")
        else:
            self.header_names = header_names
        self.row_builder = RowBuilder(functions=functions,
                                      provider_dict_list=provider_dict_list,
                                      mark_for_replacement_fields=mark_for_replacement_fields,
                                      replacement_char=replacement_char)

        assert len(self.header_names)==len(functions)+include_row_number
        self.random_fields = random_fields

    def clean_fnc_name(self, func_list):
        return [x.title().replace("_", " ") for x in func_list]

    def choose_random_fields(self):
        self.extra_headers = []
        self.extra_functions = []

        if not self.random_fields:
            return

        for i in range(self.random_fields):
            name,fn = random.choice(self.row_builder.valid_functions)
            self.extra_functions.append(fn)
            self.extra_headers.append(name)
        self.extra_headers = self.clean_fnc_name(self.extra_headers)
        return self.extra_functions

    def gen_content(self, rows):
        extra_functions = self.choose_random_fields()
        for n in range(rows):
            row = self.row_builder.gen_row(additional_functions=extra_functions)
            if self.include_row_number:
                row = [n] + row
            yield row

    def __len__(self):
        return len(self.header_names)


if __name__=='__main__':
    functions = ["name", "address", "relationship", "job", "date", "height"]

    generator = TableDataFromFaker(functions=functions)
    my_table = list(generator.gen_content(10))
    print(my_table)