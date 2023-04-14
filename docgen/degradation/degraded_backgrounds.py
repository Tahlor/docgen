import ocrodeg
# pregenerate backgrounds
# random invert/rotate and crop

def splotches(img):
    return ocrodeg.random_blotches(img, 3e-4, 1e-4)

def fibrous1(img):
    return ocrodeg.printlike_multiscale(img)

def fibrous2(img):
    return ocrodeg.printlike_fibrous(img)
