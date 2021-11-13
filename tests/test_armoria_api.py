from unittest import TestCase
from src.armoria_api import ArmoriaAPIPayload


pairs_label_payload = {
'B A lion': {'t1': 'azure', 'shield': 'heater', 'charges': [{'charge': 'lionRampant', 't': 'argent', 'p': 'e', 'size': '1.5'}]}
}


class ArmoriaAPIPayloadTest(TestCase):

    def test_get_armoria_payload(self):
        for label, payload in pairs_label_payload.items():
            assert ArmoriaAPIPayload(label.split()).get_armoria_payload() == payload
