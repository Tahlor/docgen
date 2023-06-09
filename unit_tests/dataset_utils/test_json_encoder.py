from docgen.bbox import BBox
from docgen.dataset_utils import save_json, load_json
from json import JSONEncoder, JSONDecoder
from docgen.dataset_utils import JSONEncoder as myJSONEncoder

test_dict = {'paragraphs': None, 'bbox': (0, 0, 1737, 2059), 'level': 1, 'category': 'page', 'id': (1, 0), 'parent_id': (0, 0)}
test_dict2 = {'paragraphs': None, 'bbox': BBox("ul", (0, 0, 1737, 2059)), 'level': 1, 'category': 'page', 'id': (1, 0), 'parent_id': (0, 0)}

mine = myJSONEncoder()
theirs = JSONEncoder()
y = JSONDecoder()

try:
    m = y.decode( mine.encode(test_dict))
    print(m)
except Exception as e:
    print(e)

try:
    m = y.decode( mine.encode(test_dict2))
    print(m)
except Exception as e:
    print(e)

try:
    m = y.decode( theirs.encode(test_dict))
    print(m)
except Exception as e:
    print(e)

try:
    m = y.decode( theirs.encode(test_dict2))
    print(m)
except Exception as e:
    print(e)
