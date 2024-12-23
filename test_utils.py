# test_utils.py

import unittest
import pandas as pd
from utils import append_metrics

class TestUtils(unittest.TestCase):
    def test_append_metrics(self):
        master_df = pd.DataFrame(columns=['Model', 'Perturbation_Step', 'Epoch', 'Metric_Type', 'Metric_Name', 'Metric_Value'])
        metrics = [
            {
                'Model': 'Test_Model',
                'Perturbation_Step': 'Test_Step',
                'Epoch': 1,
                'Metric_Type': 'Test_Type',
                'Metric_Name': 'Test_Metric',
                'Metric_Value': 0.95
            }
        ]
        updated_df = append_metrics(master_df, metrics)
        self.assertEqual(len(updated_df), 1)
        self.assertEqual(updated_df.iloc[0]['Model'], 'Test_Model')
        self.assertEqual(updated_df.iloc[0]['Metric_Value'], 0.95)

if __name__ == '__main__':
    unittest.main()
