import xml.etree.ElementTree as ET

tree = ET.parse('../static/sumo-output/sumo-output.xml')
elts = tree.getroot().findall('elements')
elements = []
for elt1 in elts:
    for elt2 in elt1:
        for elt in elt2:
            if elt.tag == 'pose':
                for data in elt:
                    if data.tag == 'translation':
                        translation = [float(k) for k in data.text.replace(' ', '').split(',')]
                    elif data.tag == 'rotation':
                        rotation = []
                        for t in data:
                            rotation.append([float(k) for k in t.text.replace(' ', '').split(',')])
            elif elt.tag == 'id':
                id_ = elt.text
        elements.append(['sumo-output/%s.glb' % (id_), rotation, translation])
with open('../static/data/models_data.json', 'w') as f:
    f.write("var models_data = %s" % str(elements))



