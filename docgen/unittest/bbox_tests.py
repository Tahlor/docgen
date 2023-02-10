from docgen.bbox import BBoxNGon, BBox
import numpy as np
import matplotlib.pyplot as plt

list_of_bboxes = [[0, 0, 4, 4, 8, 4], [1, 1, 5, 7], [7, 5],[3,9]]
#equal = TestCase.assertSequenceEqual

def equal(L1,L2):
    assert list(L1) == list(L2)

def plot_hull(convex_hull, points):
    """

    Args:
        convex_hull: 2D array N x 2
        points: 2D array N x 2

    Returns:

    """
    fig, ax = plt.subplots(ncols=1, figsize=(10, 3))
    ax.set_title('Convex hull')
    ax.plot(points[:, 0], points[:, 1], '.', color='k')
    ax.plot(convex_hull[:,0], convex_hull[:,1], 'o', mec='r', color='none', lw=1, markersize=10)
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    plt.show()


def convex_hull():
    convex_hull = BBoxNGon.get_convex_hull(list_of_bboxes)
    points = np.concatenate(list_of_bboxes).reshape(-1,2)
    plot_hull(convex_hull.reshape(-1,2), points)


def bbox_origin_format_consistency():
    bbox = [5,13,26,57]
    box = BBox("ul", bbox, format="XYXY", height_ll=1000)
    out = box._XYWH_ll_to_XYXY_ul(box._XYXY_ul_to_XYWH_ll([5,13,26,57], height=box.height), height=box.height)
    out2 = box._XYWH_to_XYXY(box._get_XYWH(box))
    equal(bbox, out2)
    equal(bbox, out)

def concave_hull_test():
    import matplotlib.pyplot as plt

    # 8 points defining rectangle from 0,0 to 10,10
    flattened_bbox_list_of_clockwise_points = [0, 0, 5, 0, 10, 0, 10, 5, 10, 10, 5, 10, 0, 10, 0, 5]
    shuffled = np.array(flattened_bbox_list_of_clockwise_points).reshape(-1, 2)
    np.random.shuffle(shuffled)

    m = BBoxNGon.concave_hull(shuffled)
    plt.plot(m[:, 0], m[:, 1])
    plt.plot(shuffled[:, 0], shuffled[:, 1])
    plt.show()

def bbox_ngon_consistency():
    list_of_bboxes = [[0, 0, 4, 4], [1, 1, 5, 7], [-1, 2, 2, 3]]

    all_funcs = ["invert_y_axis",
                 "swap_bbox_axes",
                 "offset_origin",
                 "get_dim",
                 ]
    bbox = [1,1,3,4]
    for func in all_funcs:
        print(func)
        ngon = BBoxNGon("ul", bbox, height_ll=1000)
        box = BBox("ul", bbox, height_ll=1000)
        fn = ngon.__getattribute__(func)()
        fb = box.__getattribute__(func)()
        print(fn, fb)
        equal(fn, fb)
    assert ngon.height == box.height
    assert ngon.width == box.width
    equal(ngon.get_maximal_box_orthogonal(list_of_bboxes), box.get_maximal_box(list_of_bboxes))


"""
Compare BBox and BBoxNGon on regular BBox to make sure they are consistent

"""

if __name__=='__main__':
    #convex_hull()
    bbox_ngon_consistency()
    bbox_origin_format_consistency()