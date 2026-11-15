
import requests
import xml.etree.ElementTree as ET

url = (
    "https://trek.nasa.gov/tiles/Mars/EQ/"
    "CTX_beta01_uncontrolled_5m_Caltech/"
    "1.0.0/WMTSCapabilities.xml"
)

response = requests.get(url, timeout=30)
response.raise_for_status()

root = ET.fromstring(response.content)

print("\n===== NASA CTX WMTS LAYERS =====")

for layer in root.iter():
    if layer.tag.endswith("Layer"):
        identifier = next(
            (e.text for e in layer
             if e.tag.endswith("Identifier")),
            None
        )

        if identifier:
            print("\nLAYER:", identifier)

        for item in layer:
            if item.tag.endswith("ResourceURL"):
                print("FORMAT:", item.attrib.get("format"))
                print("RESOURCE TYPE:", item.attrib.get("resourceType"))
                print("TILE TEMPLATE:", item.attrib.get("template"))

        for item in layer:
            if item.tag.endswith("TileMatrixSetLink"):
                for child in item:
                    if child.tag.endswith("TileMatrixSet"):
                        print("MATRIX SET:", child.text)

print("\n===== END OF NASA DIAGNOSTIC =====")
