"""Offline regression tests; no Streamlit session or OpenAI call required."""
import ast
from pathlib import Path
import unittest

SOURCE = (Path(__file__).resolve().parents[1] / 'app_fixed.py').read_text()
TREE = ast.parse(SOURCE.split('# UI\n')[0])
NODES = [n for n in TREE.body if (
    isinstance(n, (ast.FunctionDef, ast.Assign))
    or isinstance(n, ast.ImportFrom) and n.module == 'datetime'
    or isinstance(n, ast.Import) and all(a.name in {'json', 'random', 're', 'string'} for a in n.names)
)]
NS = {}
exec(compile(ast.Module(body=NODES, type_ignores=[]), '<app logic>', 'exec'), NS)


class RegressionTests(unittest.TestCase):
    def test_syntax(self):
        compile(SOURCE, 'app_fixed.py', 'exec')

    def test_blank_case_id_cannot_create_ticket(self):
        for case_id in ['', ' ', '\t\n']:
            for decision in ['Accept', 'Modify', 'Escalate']:
                with self.subTest(case_id=case_id, decision=decision):
                    result = NS['create_simulated_record'](
                        case_id, 55, 'Specialist', decision, 'Investigate',
                        'Standard', 'Troubleshoot', 'Work impact')
                    self.assertTrue(result['status'].startswith('BLOCKED'))
                    self.assertEqual(result['ticket_id'], '')

    def test_case_id_trimmed_and_audit_preserved(self):
        result = NS['create_simulated_record'](
            ' REG-04 ', 55, 'Specialist', 'Escalate', 'Investigate',
            'Standard', 'Troubleshoot', 'Work impact', 'ASSESS-REG-04')
        self.assertEqual(result['case_id'], 'REG-04')
        self.assertEqual(result['assessment_id'], 'ASSESS-REG-04')
        self.assertEqual(result['original_queue'], 'Standard')
        self.assertEqual(result['override_reason'], 'Work impact')
        self.assertTrue(result['ticket_id'].startswith('SIM-'))

    def test_contact_context(self):
        for message, expected in [
            ('I restarted the router twice', []),
            ('I restarted the router two times', []),
            ('I contacted support twice', [2]),
            ('I have contacted support four times already', [4]),
            ('I called support once and restarted the router twice', [1]),
        ]:
            with self.subTest(message=message):
                self.assertEqual(NS['extract_mentioned_contact_counts'](message), expected)
        self.assertIsNotNone(NS['detect_contact_count_discrepancy'](0, 'I contacted support four times', ''))

    def test_score_and_escalation_are_separate(self):
        result = NS['calculate_priority'](1, True, True, 'Neutral', 'Low', 'High')
        self.assertEqual(result['score'], 55)
        self.assertEqual(result['trigger'], 'High urgency')
        self.assertTrue(result['escalation_recommended'])
        routine = NS['calculate_priority'](0, False, False, 'Neutral', 'Low', 'Low')
        self.assertEqual(routine['score'], 10)
        self.assertFalse(routine['escalation_recommended'])

    def test_override_requires_reason(self):
        for decision in ['Modify', 'Escalate']:
            result = NS['create_simulated_record'](
                'REG-04', 55, 'Specialist', decision, 'Investigate',
                'Standard', 'Troubleshoot', ' ')
            self.assertTrue(result['status'].startswith('BLOCKED'))

    def test_traceable_assessment_id_is_required(self):
        result = NS['create_simulated_record'](
            'REG-01', 87, 'Specialist', 'Accept', 'Investigate',
            'Specialist', 'Investigate')
        self.assertTrue(result['status'].startswith('BLOCKED'))
        self.assertEqual(result['ticket_id'], '')

    def test_resolved_status_keywords(self):
        for status in ['working now', 'resolved', 'back to normal', 'restored', 'fixed']:
            self.assertNotIn('current status', NS['check_missing_information'](status))


if __name__ == '__main__':
    unittest.main()
