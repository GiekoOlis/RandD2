import matplotlib.pyplot as plt
import numpy as np
import pydicom
import skimage.io
import cv2
import os
import json
from PIL import Image, ImageDraw, ImageColor, ImageFont
from operator import itemgetter

import PIL

corner_points_SD = ["CRV", "CAV", "CRD", "CAD"]
corner_points_FD = ["CRVL", "CAVL", "CRVR", "CAVL"]

palette_color = {
    "C": "#ff0000",
    "T": "#ff7128",
    "L": "#ffcc00",
    "S": "#92d050",
    "F": "#00b0f0"
}

palette_white = {
    "C": "white",
    "T": "white",
    "L": "white",
    "S": "white",
    "F": "white"
}


def read_mrk_json(path_to_markdown, markdown, encoding = "utf-8"):
    """
    Read *.mrk.json file containing annotations for dicom file

    Parameters
    ----------
    path_to_markdown : str
        The file location of the *.mrk.json file
    markdown : str
        The file name including ".mrk.json" extension
    encoding : str, optional
        The encoding of *.mrk.json file (default is "utf-8")

    Returns
    -------
    dict
        contains spine element name, 
        number of points in annotation, 
        list of pairs of coordinates (x, y)
    """

    try:
        with open(os.path.join(path_to_markdown, markdown)) as f:
            data = json.load(f)
    except FileNotFoundError as fnf_error:
        print(fnf_error)
    else:
        control_points = []
        for i in data['markups'][0]['controlPoints']:
            control_points.append({
                'id': i['id'],
                'label': i['label'],
                'position': i['position']
            })
        point_set = {
            'name': markdown.split(".")[0],
            'number_of_points': len(control_points),
            'controlPoints': control_points,
        }
        return point_set


def read_all_markdowns(path_to_labels):
    """
    Read all the annotations for one case (C2-C7, Th1-Th12, L1-L5, S1, FH1, FH2)

    Parameters
    ----------
    path_to_labels: str
        Path to folder containing all annotations (26 *.mrk.json files)

    Returns
    -------
    list
        list of dictionaries    
    """

    filenames = os.listdir(path_to_labels)
    all_point_sets = []
    for file in filenames:
        all_point_sets.append(read_mrk_json(path_to_labels, file))
    
    return all_point_sets


def define_circle(p1, p2, p3):
    """
    Returns the center and radius of the circle passing the given 3 points.
    In case the 3 points form a line, returns (None, infinity).
    """
    temp = p2[0] * p2[0] + p2[1] * p2[1]
    bc = (p1[0] * p1[0] + p1[1] * p1[1] - temp) / 2
    cd = (temp - p3[0] * p3[0] - p3[1] * p3[1]) / 2
    det = (p1[0] - p2[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p2[1])
    
    if abs(det) < 1.0e-6:
        return None, np.inf

    # Center of circle
    cx = (bc*(p2[1] - p3[1]) - cd*(p1[1] - p2[1])) / det
    cy = ((p1[0] - p2[0]) * cd - (p2[0] - p3[0]) * bc) / det

    radius = np.sqrt((cx - p1[0])**2 + (cy - p1[1])**2)
    return (cx, cy), radius



def get_LUT_value(data, window, level):
    """
    Apply Window Width and Window Level parameters to dicom pixel array

    Parameters
    ----------
    data: numpy array
        dicom pixel array
    window: int
        window center
    level: int
        window width

    Returns
    -------
    numpy array
        processed dicom pixel array
    """
    return np.piecewise(data,
                        [data <= (level - 0.5 - (window - 1) / 2),
                         data > (level - 0.5 + (window - 1) / 2)],
                        [0, 255, lambda data: ((data - (level - 0.5)) /
                         (window - 1) + 0.5) * (255 - 0)])


# TODO: add window and level parameters as input, if they r not None then use input, if they r None try to use from dicom
def get_PIL_image(dataset):
    """Get Image object from Python Imaging Library(PIL)"""
    if ('PixelData' not in dataset):
        raise TypeError("Cannot show image -- DICOM dataset does not have "
                        "pixel data")
    # can only apply LUT if these window info exists
    if ('WindowWidth' not in dataset) or ('WindowCenter' not in dataset):
        # print(dataset.PhotometricInterpretation)
        bits = dataset.BitsAllocated
        samples = dataset.SamplesPerPixel
        if bits == 8 and samples == 1:
            mode = "L"
        elif bits == 8 and samples == 3:
            mode = "RGB"
        elif bits == 16:
            # not sure about this -- PIL source says is 'experimental'
            # and no documentation. Also, should bytes swap depending
            # on endian of file and system??
            mode = "I;16"
        else:
            raise TypeError("Don't know PIL mode for %d BitsAllocated "
                            "and %d SamplesPerPixel" % (bits, samples))

        # PIL size = (width, height)
        size = (dataset.Columns, dataset.Rows)
        print(mode)
        # Recommended to specify all details
        # by http://www.pythonware.com/library/pil/handbook/image.htm
        im = PIL.Image.frombuffer(mode, size, dataset.pixel_array,
                                  "raw", mode, 0, 1)

    else:
        ew = dataset['WindowWidth']
        ec = dataset['WindowCenter']
        ww = int(ew.value[0] if ew.VM > 1 else ew.value)
        wc = int(ec.value[0] if ec.VM > 1 else ec.value)
        image = get_LUT_value(dataset.pixel_array, ww, wc)
        # Convert mode to L since LUT has only 256 values:
        #   http://www.pythonware.com/library/pil/handbook/image.htm
        im = PIL.Image.fromarray(image).convert('L')

    return im


def create_filled_mask(path_to_orig_image, all_point_sets, path_to_output, colored = False, spacing = None, draw_FH=True):
    output = os.path.join(path_to_output, path_to_orig_image.split("\\")[-1].split(".")[0] + ".png")
    dicom = pydicom.dcmread(path_to_orig_image)
    input = get_PIL_image(dicom)
    if spacing == None:
        spacing = dicom.ImagerPixelSpacing
    width, height = input.size

    if colored == False:
        mask = Image.new('1', (width, height), 'black')
        palette = palette_white
    else:
        mask = Image.new('RGB', (width, height), 'black')
        palette = palette_color
    draw = ImageDraw.Draw(mask)
    for k, markdown in enumerate(all_point_sets):
        if "FH" in markdown['name']:
            if draw_FH:
                coordinates = [(float(i['position'][0]), float(i['position'][1])) for i in markdown['controlPoints']]
                c, r = define_circle(coordinates[0], coordinates[1], coordinates[2])
                c, r = (c[0] / spacing[0], c[1] / spacing[1]), r / spacing[0]
                draw.point((c[0], c[1]), fill=palette["F"])
                draw.ellipse([(c[0]-r, c[1]-r), (c[0]+r, c[1]+r)],  fill=palette["F"], outline = 'white', width=1)
        else:
            coordinates = [(float(i['position'][0] / spacing[0]), float(i['position'][1]) / spacing[1]) for i in markdown['controlPoints']]
            draw.polygon(tuple(coordinates), fill=palette[markdown['name'][0]], outline = 'white', width=1)
            print(coordinates)
    mask.save(output)
        
        
def create_point_mask(path_to_orig_image, all_point_sets, path_to_output, colored = False, spacing = None, point_size = 10, draw_FH=True, dim="_SD"):
    if dim == "_SD":
        point_names = corner_points_SD
    elif dim == "_FD":
        point_names = corner_points_FD


    output = os.path.join(path_to_output, path_to_orig_image.split("\\")[-1].split(".")[0] + ".png")
    dicom = pydicom.dcmread(path_to_orig_image)
    if spacing == None:
        spacing = dicom.ImagerPixelSpacing

    input = get_PIL_image(dicom)
    width, height = input.size
    if colored == False:
        mask = Image.new('1', (width, height), 'black')
        palette = palette_white
    else:
        mask = Image.new('RGB', (width, height), 'black')
        palette = palette_color
    draw = ImageDraw.Draw(mask)
    for k, markdown in enumerate(all_point_sets):
        if "FH" in markdown['name']:
            if draw_FH:
                coordinates = [(float(i['position'][0]), float(i['position'][1])) for i in markdown['controlPoints']]
                c, r = define_circle(coordinates[0], coordinates[1], coordinates[2])
                c, r = (c[0] / spacing[0], c[1] / spacing[1]), r / spacing[0]
                draw.ellipse([([0]-r, c[1]-r), (c[0]+r, c[1]+r)], outline = 'white', width=1)
                draw.ellipse([(c[0]-point_size, c[1]-point_size), (c[0]+point_size, c[1]+point_size)], fill='yellow', outline = 'yellow', width=1)
                draw.point((c[0], c[1]), fill='black')
        else:
            points = [x for x in markdown['controlPoints'] if x['label'] in point_names]
            coordinates = [(float(i['position'][0] / spacing[0]), float(i['position'][1]) / spacing[1]) for i in points]
            for point in coordinates:
                draw.ellipse([(point[0]-point_size, point[1]-point_size), (point[0]+point_size, point[1]+point_size)], fill=palette[markdown['name'][0]], width=1)
    
    mask.save(output)

def read_mask(mask_name):
    mask = (skimage.io.imread(mask_name)[:,:]==255).astype(np.uint8)*255
    mask = (mask > 0).astype(np.uint8)
    return mask


def read_image(img_name, mask_name=None, channels3 = True, return_dicom = False):
    dicom = pydicom.dcmread(img_name)
    img = dicom.pixel_array
    img = (img / img.max())*255
    img = np.uint8(img)
    img = np.stack((img,)*3, axis=-1) if channels3 else img
    res = [img]
    mask = read_mask(mask_name) if mask_name is not None else None
    # mask = img = np.stack((mask,)*3, axis=-1) if channels3 else mask
    dicom = dicom if return_dicom is True else None
    res.append(mask)
    res.append(dicom)
    return res


def read_image_1(img_name, mask_name=None, channels3 = True, return_dicom = False):
    dicom = pydicom.dcmread(img_name)
    img = dicom.pixel_array
    # img = (img / img.max())*255
    # img = np.uint8(img)
    img = np.stack((img,)*3, axis=-1) if channels3 else img
    res = [img]
    mask = read_mask(mask_name) if mask_name is not None else None
    # mask = img = np.stack((mask,)*3, axis=-1) if channels3 else mask
    dicom = dicom if return_dicom is True else None
    res.append(mask)
    res.append(dicom)
    return res


def make_blending(img_path, mask_path, alpha=0.5):
    img, mask, _ = read_image(img_path, mask_path)
    colors = np.array([[0,0,0], [255,0,0]], np.uint8)
    return (img*alpha + colors[mask.astype(np.int32)]*(1. - alpha)).astype(np.uint8)



def show_images_with_mask(image_path, mask_path_point, mask_path_fill, alpha=0.5):
    plt.figure(figsize=(20, 14))
    plt.subplot(1, 3, 1)
    orig, _m, _d = read_image(image_path)
    plt.imshow(orig)
    plt.subplot(1, 3, 2)
    blend = make_blending(image_path, mask_path_fill, alpha)
    plt.imshow(blend)
    plt.subplot(1, 3, 3)
    blend = make_blending(image_path, mask_path_point, alpha)
    plt.imshow(blend)




def show_markdown(path, case, corner_points=False, body=False, bbox=False, body_fill=False, dim = "_SD", palette = {
    "C": "#ff0000",
    "T": "#ff7128",
    "L": "#ffcc00",
    "S": "#92d050",
    "F": "#00b0f0",
    "outline": "white",
    "fill": "white",
    "points": "blue"
}):
    """Shows all aforementioned markdowns

    Parameters
    ----------
    path : str
        The file location of the dataset
    case : str
        The case name (same in "dicom" and "labels" folders)
    corner_points : bool, optional
        A flag used to draw the corner points labeling (default is False)
    body : bool, optional
        A flag used to draw the countours of the vertebrae (default is False)
    body_fill : bool, optional
        A flag used to draw the bounding boxes around vertebrae (default is False)
    palette : dict, optional
        A flag used to fill the vertebrae contours (default is False)

    Returns
    -------
    PIL.Image.Image
        PIL image object 
    """

    data_coordinates = []
    path_to_orig_image = os.path.join(*[path, "dicom", case + ".dcm"])
    spacing = None

    dicom = pydicom.dcmread(path_to_orig_image)
    input = Image.fromarray(dicom.pixel_array)
    input = get_PIL_image(dicom)
    input = input.convert('RGB')
    print(input)
    plt.figure(figsize=(50, 30))
    if spacing == None:
        spacing = dicom.ImagerPixelSpacing
    page_width, page_height = input.size
    markdowns = read_all_markdowns(os.path.join(*[path, "labels", case]))
    draw = ImageDraw.Draw(input)
    for k, markdown in enumerate(markdowns):
        filename = case + ".png"

        if "FH"  not in markdown['name']:
            category = 'Vertebrae' if "C2" not in markdown['name'] else 'C2'
            coordinates = [(float(i['position'][0]) / spacing[0], float(i['position'][1]) / spacing[1]) for i in markdown['controlPoints']]

            min_x, max_x = min(coordinates, key=itemgetter(0))[0], max(coordinates, key=itemgetter(0))[0]
            min_y, max_y = min(coordinates, key=itemgetter(1))[1], max(coordinates, key=itemgetter(1))[1]

        else:
            category = 'FermutHead'
            coordinates = [(float(i['position'][0]), float(i['position'][1])) for i in markdown['controlPoints']]
            c, r = define_circle(coordinates[0], coordinates[1], coordinates[2])
            c, r = (c[0] / spacing[0], c[1] / spacing[1]), r / spacing[0]

            min_x, max_x = c[0] - r, c[0] + r
            min_y, max_y = c[1] - r, c[1] + r
            coordinates = []
        
        category1 = markdown['name']
        width = max_x - min_x
        height = max_y - min_y
        x, y = int(min_x), int(min_y)
        w, h = int(width), int(height)

        coordinates2 = [(min_x - 2, min_y - 2), (min_x - 2, min_y + h + 2), (min_x + w + 2, min_y + h + 2), (min_x + w + 2, min_y - 2)]

        if body and coordinates:
            draw.polygon(tuple(coordinates), outline=palette['outline'], width=1)

        if body_fill and coordinates:
            draw.polygon(tuple(coordinates), outline=palette['outline'], fill=palette['fill'], width=1)  
        if dim == "_SD":
            points = [x for x in markdown['controlPoints'] if x['label'] in corner_points_SD]
        elif dim == "_FD":
            points = [x for x in markdown['controlPoints'] if x['label'] in corner_points_FD]
        coordinates_points = [(float(i['position'][0] / spacing[0]), float(i['position'][1]) / spacing[1]) for i in points]
        point_size = 5
        if corner_points:
            for point in coordinates_points:
                draw.ellipse([(point[0]-point_size, point[1]-point_size), (point[0]+point_size, point[1]+point_size)], fill=palette['points'], width=1)

        if bbox:
            draw.polygon(tuple(coordinates2), outline = palette[markdown['name'][0]], width=2)
            font = ImageFont.truetype('arial', size=22)
            draw.text(
                (min_x + w + 30, min_y),  # Coordinates
                case + " " + markdown['name'],  # Text
                palette[markdown['name'][0]],  # Color
                font
            )

        row1 = [filename, page_height, page_width, category, category1, x, y, w, h]
        data_coordinates.append(row1)

    return input, data_coordinates    