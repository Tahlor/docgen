import random
import datetime

date_separators = ['/', '-', '.', ' ']
date_formats_with_hyphens = ["%Y-%m-%d", "%m-%d-%Y", "%d-%m-%Y", "%Y-%d-%m", "%m-%Y-%d", "%d-%Y-%m"]
def random_number_generator():
    # Randomly choose the type of number to generate
    number_type = random.choice(['integer', 'float', 'date', 'formatted', 'with_symbols'])
    length = random.randint(0, 8)

    if number_type == 'integer':
        if random.random() < 0.5:
            return random.randint(0, 10**length)
        else:
            return f"{random.randint(0, 10**length):,}"
    elif number_type == 'float':
        return round(random.uniform(0, 10**length), 2)
    elif number_type == 'date':
        start_date = datetime.date(1600, 1, 1)
        end_date = datetime.date(2100, 12, 31)
        random_date = start_date + datetime.timedelta(days=random.randint(0, (end_date - start_date).days))
        random_format = random.choice(date_formats_with_hyphens)
        random_symbol = random.choice(date_separators)
        return random_date.strftime(random_format.replace('-', random_symbol))
    elif number_type == 'formatted':
        return f"{random.randint(0, 10000):,}"
    elif number_type == 'with_symbols':
        number = random.randint(0, 10000)
        symbol = random.choice(['#', '$', '%', '&', '@'])
        format_style = random.choice(['prefix', 'suffix', 'parentheses'])
        if format_style == 'prefix':
            return f"{symbol}{number}"
        elif format_style == 'suffix':
            return f"{number}{symbol}"
        elif format_style == 'parentheses':
            return f"({number})"

# Example usage
for _ in range(5):
    print(random_number_generator())
