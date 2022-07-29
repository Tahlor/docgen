from faker.ancestry.custom_row import RowBuilder

class TableDataFromFaker:
    """ Generate content
    """
    def __init__(self, functions,
                 header_names=None,
                 include_row_number=True,
                 provider_dict_list=None,
                 mark_for_replacement_fields=None,
                 replacement_char=None):
        self.include_row_number = include_row_number
        if header_names is None:
            self.header_names = [x.title().replace("_"," ") for x in functions]
            if include_row_number:
                self.header_names.insert(0, "Row")
        else:
            self.header_names = header_names
        self.row_builder = RowBuilder(functions=functions,
                                      provider_dict_list=provider_dict_list,
                                      mark_for_replacement_fields=mark_for_replacement_fields,
                                      replacement_char=replacement_char)

        assert len(self.header_names)==len(functions)+include_row_number

    def gen_content(self, rows):
        for n in range(rows):
            row = self.row_builder.gen_row()
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