import os
from glob import glob
import cv2
import numpy as np
from tqdm import tqdm
from datetime import datetime
import lxml.etree as ET

_ns = {'p': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'}
output_folder='crop_csg18_prediction_xml_eme/'
#os.makedirs('crop_cb55_prediction_xml')
polygon_labels_dir= 'crop_csg18_polygon_labels_eme/polygon_labels_eme/'
original_pages_dir= 'CSG18/'
crop_pages_dir='crop_csg18/'

def get_page_filename(image_filename: str) -> str:
    return os.path.join(os.path.dirname(image_filename),
                        '{}.xml'.format(os.path.basename(image_filename)[:-4]))


def get_basename(image_filename: str) -> str:
    directory, basename = os.path.split(image_filename)
    return '{}'.format( basename)


def save_and_resize(img: np.array, filename: str, size=None) -> None:
    if size is not None:
        h, w = img.shape[:2]
        resized = cv2.resize(img, (int(w*size), int(h*size)),
                             interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(filename, resized)
    else:
        cv2.imwrite(filename, img)


def xml_to_coordinates(t):
    result = []
    for p in t.split(' '):
        values = p.split(',')
        assert len(values) == 2
        x, y = int(float(values[0])), int(float(values[1]))
        result.append((x,y))
    result=np.array(result)
    return result

def coordinates(cnt):
    coords=''
    for i in range(len(cnt[0])):
        coord=str(cnt[0][i][0][0])+','+str(cnt[0][i][0][1])+' '
        coords=coords+coord
    return coords

original_path_list = glob('{}/*.jpg'.format(original_pages_dir))
for original_path in original_path_list:
    xmlns = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2018-07-15"
    xsi = "http://www.w3.org/2001/XMLSchema-instance"
    schemaLocation = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2017-07-15 http://schema.primaresearch.org/PAGE/gts/pagecontent/2017-07-15/pagecontent.xsd"

    PcGts = ET.Element("{" + xmlns + "}PcGts",
                       attrib={"{" + xsi + "}schemaLocation": schemaLocation},
                       nsmap={'xsi': xsi, None: xmlns})

    Metadata = ET.SubElement(PcGts, 'Metadata')
    Creator = ET.SubElement(Metadata, 'Creator')
    Creator.text = 'Berat'
    Metadata.append(Creator)
    Created = ET.SubElement(Metadata, 'Created')
    Created.text = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
    Metadata.append(Created)
    LastChange = ET.SubElement(Metadata, 'LastChange')
    LastChange.text = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
    Metadata.append(LastChange)
    PcGts.append(Metadata)

    original_page_name=get_basename(original_path)
    original_page = cv2.imread(original_path, 1)
    rows, cols, ch = original_page.shape
    Page = ET.SubElement(PcGts, 'Page')
    Page.set('imageFilename', original_page_name)
    Page.set('imageWidth', str(cols))
    Page.set('imageHeight', str(rows))

    TextRegion = ET.SubElement(Page, 'TextRegion')
    TextRegion.set('id', 'region_textline')
    TextRegion.set('custom', '0')
    Coords = ET.SubElement(TextRegion, 'Coords')
    page_filename = get_page_filename(original_page_name)
    xml_page = ET.parse(original_pages_dir+'page/'+page_filename)
    page_elements = xml_page.find('p:Page', _ns)
    regions= page_elements.findall('p:TextRegion', _ns)
    coords=regions[0].find('p:Coords', _ns)
    points = coords.attrib['points']
    Coords.set('points', points)
    TextRegion.append(Coords)


    polygon_labels = cv2.imread(polygon_labels_dir + original_page_name[:-4] + '.png', 0)
    original_polygon_labels = np.zeros((rows, cols), dtype=np.uint8)
    cnt=xml_to_coordinates(points)
    (x,y,w,h) = cv2.boundingRect(cnt)
    original_polygon_labels[y:y+h, x:x+w] = polygon_labels

    labels = np.unique(original_polygon_labels)[1:]
    textlineid = 0
    for tlabel in labels:
        textline = np.zeros((rows, cols), dtype=np.uint8)
        textline[original_polygon_labels == tlabel] = 255
        tcontours, hierarchy = cv2.findContours(textline, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        tcoords = coordinates(tcontours)
        TextLine = ET.SubElement(TextRegion, 'TextLine')
        TextLine.set('id', 'textline_' + str(textlineid))
        TextLine.set('custom', '0')
        textlineid = textlineid + 1
        Coords = ET.SubElement(TextLine, 'Coords')
        Coords.set('points', tcoords)
        TextLine.append(Coords)
        TextRegion.append(TextLine)

    Page.append(TextRegion)

    PcGts.append(Page)

    mydata = ET.tostring(PcGts, pretty_print=True, encoding='utf-8', xml_declaration=True)
    myfile = open(output_folder + original_page_name[:-4] + '.xml', "w")
    myfile.write(mydata.decode("utf-8"))
    myfile.close()

