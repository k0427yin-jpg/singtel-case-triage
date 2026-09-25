import unittest
from unittest.mock import Mock, patch
from test_regression import NS


class AssessmentStateTests(unittest.TestCase):
    def setUp(self):
        self.inputs = dict(case_id='REG-01', prior_contacts=4, unresolved=True,
                           work_impact=True, history_text='Previous line test was scheduled.',
                           case_text='I contacted support four times. I am frustrated.')
        self.state = dict(result={'summary': 'old'}, priority={'score': 87},
                          record={'ticket_id': 'OLD-TICKET'}, assessed_inputs=dict(self.inputs),
                          assessment_run=1)
        self.model = Mock(return_value=dict(intent='Investigate', complexity='High',
                          urgency='High', sentiment='Frustrated', summary='Still unresolved.'))
        self.patch = patch.dict(NS, {'classify_case': self.model})
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def assert_cleared(self):
        for key in ['result','priority','record','assessed_inputs']:
            self.assertIsNone(self.state[key], key)

    def test_every_input_edit_clears_outputs(self):
        for key, value in dict(case_id='OTHER', prior_contacts=1, unresolved=False,
                              work_impact=False, history_text='changed', case_text='changed').items():
            with self.subTest(field=key):
                state=dict(self.state)
                edited={**self.inputs,key:value}
                self.assertTrue(NS['invalidate_changed_inputs'](state,edited))
                for output in ['result','priority','record','assessed_inputs']:
                    self.assertIsNone(state[output])
        self.model.assert_not_called()

    def test_unchanged_inputs_preserve_result(self):
        self.assertFalse(NS['invalidate_changed_inputs'](self.state,self.inputs))
        self.assertEqual(self.state['record']['ticket_id'],'OLD-TICKET')

    def test_empty_input_removes_ticket_and_never_calls_ai(self):
        self.inputs.update(case_text='',history_text='')
        self.assertIn('Not enough information',NS['run_assessment'](self.state,self.inputs,'test'))
        self.assert_cleared();self.model.assert_not_called()

    def test_conflicts_both_directions_before_api(self):
        for count, text in [(4,'I contacted support twice.'),(0,'I contacted support four times.')]:
            with self.subTest(count=count):
                self.inputs.update(prior_contacts=count,case_text=text)
                self.assertIn('No AI call was made',NS['run_assessment'](self.state,self.inputs,'test'))
                self.assert_cleared()
                event=self.state['validation_event']
                self.assertEqual(event['status'],'Blocked before API call')
                self.assertEqual(event['case_id'],self.inputs['case_id'])
                self.assertTrue(event['validation_id'].startswith('VALIDATE-'))
        self.model.assert_not_called()

    def test_history_conflict_before_api(self):
        self.inputs.update(case_text='Please investigate.',history_text='I contacted support twice.')
        self.assertIn('Discrepancy',NS['run_assessment'](self.state,self.inputs,'test'))
        self.assert_cleared();self.model.assert_not_called()

    def test_blank_ids_and_missing_key(self):
        for case_id in ['', '   ']:
            self.inputs['case_id']=case_id
            self.assertIn('Case ID',NS['run_assessment'](self.state,self.inputs,'test'))
            self.assert_cleared()
        self.inputs['case_id']='REG-01'
        self.assertIn('API key',NS['run_assessment'](self.state,self.inputs,''))
        self.assert_cleared();self.model.assert_not_called()

    def test_api_failure_clears_prior_success(self):
        self.model.side_effect=RuntimeError('API unavailable')
        self.assertIn('failed',NS['run_assessment'](self.state,self.inputs,'test'))
        self.assert_cleared()

    def test_invalid_response_does_not_commit_partial_results(self):
        self.model.return_value={'summary':'Only a summary'}
        self.assertIn('failed',NS['run_assessment'](self.state,self.inputs,'test'))
        self.assert_cleared()

    def test_success_is_atomic_and_clears_old_ticket(self):
        self.assertIsNone(NS['run_assessment'](self.state,self.inputs,'test'))
        self.assertEqual(self.state['priority']['score'],87)
        self.assertEqual(self.state['assessment_run'],2)
        self.assertIsNone(self.state['record'])
        self.assertEqual(self.state['assessed_inputs'],self.inputs)
        self.assertFalse(self.state['assessment_invalidated'])
        self.model.assert_called_once()
        sent=self.model.call_args.args[0]
        self.assertIn(self.inputs['history_text'],sent)
        self.assertIn(self.inputs['case_text'],sent)

    def test_router_twice_does_not_block(self):
        self.inputs.update(prior_contacts=1,case_text='I restarted the router twice. It is still down.')
        self.model.return_value.update(complexity='Medium',sentiment='Neutral')
        self.assertIsNone(NS['run_assessment'](self.state,self.inputs,'test'))
        self.assertEqual(self.state['priority']['score'],55)
        self.assertEqual(self.state['priority']['trigger'],'High urgency')

    def test_processing_evidence_is_bound_to_successful_call(self):
        self.model.return_value["_api_evidence"] = {
            "response_id": "test-response", "returned_model": "test-model",
            "raw_response": '{"summary":"Still unresolved."}',
        }
        self.assertIsNone(NS['run_assessment'](self.state,self.inputs,'secret-test-key'))
        evidence=self.state['processing_details']
        self.assertEqual(evidence['case_context'],self.model.call_args.args[0])
        self.assertEqual(evidence['response_id'],'test-response')
        self.assertEqual(evidence['returned_model'],'test-model')
        self.assertNotIn('missing_info',evidence['model_output'])
        self.assertNotIn('_api_evidence',evidence['model_output'])
        self.assertNotIn('secret-test-key',str(evidence))
        self.assertTrue(evidence['assessment_id'].startswith('ASSESS-'))
        NS['invalidate_changed_inputs'](self.state,{**self.inputs,'case_id':'changed'})
        self.assertIsNone(self.state['processing_details'])

    def test_failed_retry_removes_processing_evidence(self):
        NS['run_assessment'](self.state,self.inputs,'test')
        self.inputs.update(case_text='',history_text='')
        NS['run_assessment'](self.state,self.inputs,'test')
        self.assertIsNone(self.state['processing_details'])

    def test_record_retains_score_unit_and_version(self):
        result=NS['create_simulated_record'](
            'REG-01',87,'Specialist','Accept','Investigate','Specialist','Investigate',
            assessment_id='ASSESS-REG-01')
        self.assertEqual(result['score_unit'],'rule points')
        self.assertEqual(result['max_rule_points'],87)
        self.assertEqual(result['prototype_version'],NS['APP_VERSION'])
        self.assertEqual(result['assessment_id'],'ASSESS-REG-01')

if __name__=='__main__':unittest.main()
