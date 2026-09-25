import hashlib
import unittest

from test_regression import NS


class WorkspaceRequirementTests(unittest.TestCase):
    def test_model_prompt_and_scoring_contract_are_unchanged(self):
        prompt = NS['SYSTEM_PROMPT']
        self.assertEqual(len(prompt), 3165)
        self.assertEqual(
            hashlib.sha256(prompt.encode()).hexdigest(),
            '12044f34b95d9aac0b24f63f08674859f80f77db2bcca593bff377dc7e74da18',
        )
        matrix = [
            ((4, True, True, 'Frustrated', 'High', 'High'), 87,
             'High urgency; High complexity; Priority score > 70'),
            ((1, True, True, 'Neutral', 'Medium', 'High'), 55, 'High urgency'),
            ((3, True, False, 'Neutral', 'High', 'Low'), 50, 'High complexity'),
            ((0, False, False, 'Positive', 'Low', 'Low'), 10, 'None'),
        ]
        for arguments, score, trigger in matrix:
            with self.subTest(arguments=arguments):
                result = NS['calculate_priority'](*arguments)
                self.assertEqual(result['score'], score)
                self.assertEqual(result['trigger'], trigger)
        self.assertEqual(NS['MAX_RULE_POINTS'], 87)

    def test_scenario_library_prefills_inputs_only(self):
        scenarios = NS['CASE_SCENARIOS']
        prefixes = ('A ·', 'B ·', 'C ·', 'D ·', 'E1 ·', 'E2 ·', 'F ·', 'G ·', 'H ·')
        self.assertTrue(all(any(name.startswith(prefix) for name in scenarios) for prefix in prefixes))
        expected_fields = {
            'case_id', 'prior_contacts', 'unresolved', 'work_impact',
            'history_text', 'case_text',
        }
        forbidden_fields = {
            'result', 'score', 'route', 'complexity', 'urgency', 'sentiment',
            'ticket_id', 'escalation_recommended',
        }
        case_ids = []
        for name, inputs in scenarios.items():
            with self.subTest(name=name):
                self.assertEqual(set(inputs), expected_fields)
                self.assertFalse(set(inputs) & forbidden_fields)
                case_ids.append(inputs['case_id'])
        self.assertEqual(len(case_ids), len(set(case_ids)))

    def test_contradiction_and_router_twice_templates(self):
        scenarios = NS['CASE_SCENARIOS']
        conflict = next(value for name, value in scenarios.items() if name.startswith('F ·'))
        self.assertIsNotNone(NS['detect_contact_count_discrepancy'](
            conflict['prior_contacts'], conflict['case_text'], conflict['history_text']))
        router = next(value for name, value in scenarios.items() if name.startswith('G ·'))
        self.assertEqual(NS['extract_mentioned_contact_counts'](router['case_text']), [])
        self.assertIsNone(NS['detect_contact_count_discrepancy'](
            router['prior_contacts'], router['case_text'], router['history_text']))

    def test_long_term_case_states_limitation_without_new_rule(self):
        long_case = next(
            value for name, value in NS['CASE_SCENARIOS'].items() if name.startswith('H ·'))
        self.assertFalse(long_case['work_impact'])
        self.assertTrue(NS['has_meaningful_content'](long_case['case_text']))
        self.assertIsNone(NS['detect_contact_count_discrepancy'](
            long_case['prior_contacts'], long_case['case_text'], long_case['history_text']))
        limitation = NS['LONG_TERM_LIMITATION']
        self.assertIn('complete long-term contact record', limitation)
        self.assertIn('low score or Standard route does not prove', limitation)

    def test_review_note_extraction_is_non_scoring_reference(self):
        notes = NS['extract_operational_notes'](
            'Restart completed. A line test was scheduled. Customer called later.')
        self.assertIn('Restart completed', notes['troubleshooting'])
        self.assertIn('line test was scheduled', notes['pending_action'])

    def test_officer_final_statuses(self):
        record = {'ticket_id': 'SIM-1'}
        cases = [
            ('Recommendation accepted', record, 'Recommendation accepted'),
            ('Recommendation modified', record, 'Recommendation modified'),
            ('Escalated for specialist review', record, 'Escalated for specialist review'),
            ('Assessment blocked', None, 'Assessment blocked'),
        ]
        for decision_state, current_record, expected in cases:
            with self.subTest(decision_state=decision_state):
                status = NS['officer_final_status']({
                    'decision_state': decision_state,
                    'record': current_record,
                    'result': {'summary': 'Current'} if decision_state != 'Assessment blocked' else None,
                })
                self.assertEqual(status['label'], expected)
        waiting = NS['officer_final_status']({
            'decision_state': 'Awaiting officer decision', 'record': None,
            'result': {'summary': 'Current'},
        })
        self.assertEqual(waiting['label'], 'Awaiting officer decision')
        self.assertEqual(waiting['detail'], 'No simulated record created.')

    def test_record_and_customer_status_share_traceable_ids(self):
        record = NS['create_simulated_record'](
            'WEB-01', 55, 'Specialist broadband escalation queue', 'Escalate',
            'Route for escalation and further investigation.',
            'Standard broadband support queue', 'Provide guidance.',
            'Work calls remain affected.', 'ASSESS-WEB-01')
        self.assertEqual(record['assessment_id'], 'ASSESS-WEB-01')
        self.assertTrue(record['ticket_id'].startswith('SIM-'))
        state = {
            'customer_request': {'case_id': 'WEB-01'},
            'assessed_inputs': {'case_id': 'WEB-01'},
            'result': {'summary': 'Current'},
            'record': record,
            'decision_state': 'Escalated for specialist review',
        }
        status = NS['customer_status_from_state'](state)
        self.assertEqual(status['stage'], 'Specialist review arranged')
        self.assertEqual(status['ticket_id'], record['ticket_id'])

    def test_no_decision_never_creates_ticket(self):
        result = NS['create_simulated_record'](
            'REG-01', 87, 'Specialist', 'None', 'Investigate',
            'Specialist', 'Investigate', assessment_id='ASSESS-REG-01')
        self.assertTrue(result['status'].startswith('BLOCKED'))
        self.assertEqual(result['ticket_id'], '')

    def test_human_decision_guardrails(self):
        specialist = NS['calculate_priority'](4, True, True, 'Frustrated', 'High', 'High')
        specialist_queue, specialist_action = NS['recommended_route'](specialist)
        accepted = NS['validate_officer_decision'](
            specialist, 'Accept', specialist_queue, specialist_action, '')
        self.assertTrue(accepted['valid'])
        self.assertFalse(NS['validate_officer_decision'](
            specialist, 'Escalate', specialist_queue, specialist_action, '')['valid'])

        standard = NS['calculate_priority'](0, False, False, 'Neutral', 'Low', 'Low')
        standard_queue, standard_action = NS['recommended_route'](standard)
        self.assertFalse(NS['validate_officer_decision'](
            standard, 'Modify', standard_queue, standard_action, 'Reason')['valid'])
        self.assertFalse(NS['validate_officer_decision'](
            standard, 'Modify', 'Revised queue', standard_action, '')['valid'])
        modified = NS['validate_officer_decision'](
            standard, 'Modify', 'Revised queue', standard_action, 'Officer review')
        self.assertTrue(modified['valid'])
        self.assertFalse(NS['validate_officer_decision'](
            standard, 'Escalate', standard_queue, standard_action, '')['valid'])
        escalated = NS['validate_officer_decision'](
            standard, 'Escalate', standard_queue, standard_action, 'New evidence')
        self.assertTrue(escalated['valid'])
        self.assertEqual(escalated['final_queue'], 'Specialist broadband escalation queue')


if __name__ == '__main__':
    unittest.main()
