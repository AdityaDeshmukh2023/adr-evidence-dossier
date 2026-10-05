from dataclasses import replace
from itertools import product

from adr_system.engine import analyze_medications, analyze_structured
from adr_system.models import Candidate, Medication
from adr_system.terminology import exact
from adr_system.uncertainty import candidate_assessments, clear_candidate_cache, candidate_cache_info


def candidate(name):
    return exact(name)[0]


def ambiguous(ident, *names):
    return Medication('unreadable', None, mention_id=ident,
                      candidates=tuple(candidate(n) for n in names))


def test_conditional_findings_do_not_create_resolved_alerts():
    result = analyze_medications('Warfarln\nAspirin')
    assert not result.alerts
    assessment = candidate_assessments(result.medications)[0]
    assert 'high' in assessment['possible_states']
    assert assessment['scope'] == 'retained_candidates_only'
    assert any(o['records'] for o in assessment['outcomes'])


def test_stable_severity_does_not_confirm_a_source_record():
    meds = [ambiguous('m1', 'warfarin', 'apixaban'),
            replace(analyze_medications('aspirin').medications[0], mention_id='m2')]
    assessment = candidate_assessments(meds)[0]
    assert assessment['status'] == 'stable'
    assert assessment['stable_states'] == ['high']
    assert not assessment['stable_record_ids']
    assert len({r['source_record_id'] for o in assessment['outcomes'] for r in o['records']}) == 2


def test_candidate_join_equals_exhaustive_small_prescription():
    meds = [ambiguous('m1', 'warfarin', 'metformin'),
            ambiguous('m2', 'aspirin', 'amoxicillin'),
            ambiguous('m3', 'apixaban', 'acetaminophen')]
    expected = set()
    for interpretation in product(*(m.candidates for m in meds)):
        resolved = [replace(m, ingredient_ids=c.ingredient_ids, ingredient_names=c.names,
                            normalized=True, status='exact') for m, c in zip(meds, interpretation)]
        expected.update((tuple(sorted(a.ingredient_ids)), a.severity)
                        for a in analyze_structured(resolved, include_review=False).alerts)
    actual = {(tuple(r['ingredient_ids']), r['severity'])
              for assessment in candidate_assessments(meds)
              for outcome in assessment['outcomes'] for r in outcome['records']}
    assert actual == expected


def test_unusable_and_excluded_entries_and_mutation_safe_cache():
    unknown = Medication('ZZZ', None, mention_id='m1')
    med = replace(analyze_medications('aspirin').medications[0], mention_id='m2')
    assert candidate_assessments([unknown, med])[0]['status'] == 'unusable_identity'
    assert candidate_assessments([replace(unknown, status='excluded'), med]) == []
    clear_candidate_cache()
    first = candidate_assessments([med, replace(med, mention_id='m3')])
    first[0]['outcomes'][0]['states'].append('corruption')
    second = candidate_assessments([med, replace(med, mention_id='m3')])
    assert second[0]['possible_states'] == ['same_ingredient']
    assert candidate_cache_info().hits == 1


def test_single_uncertain_combination_product_retains_internal_findings():
    aspirin, warfarin, apixaban = (candidate(name) for name in ('aspirin', 'warfarin', 'apixaban'))
    choices = tuple(Candidate(tuple(sorted(aspirin.ingredient_ids + other.ingredient_ids)),
                              aspirin.names + other.names, .8, 'fixture')
                    for other in (warfarin, apixaban))
    medication = Medication('unreadable combination', None, mention_id='m1', candidates=choices)
    result = analyze_structured([medication])
    assert not result.alerts
    assessment = result.candidate_assessments[0]
    assert assessment['scope'] == 'within_entry_retained_candidates_only'
    assert assessment['status'] == 'stable'
    assert assessment['stable_states'] == ['high']
    assert not assessment['stable_record_ids']
    actual = {r['source_record_id'] for o in assessment['outcomes'] for r in o['records']}
    expected = set()
    for choice in choices:
        resolved = replace(medication, ingredient_ids=choice.ingredient_ids,
                           ingredient_names=choice.names, normalized=True, status='exact')
        expected.update(a.source_record_id for a in analyze_structured([resolved]).alerts)
    assert actual == expected and len(actual) == 2


def test_combination_candidate_and_single_ingredient_candidate_are_conditional():
    aspirin, warfarin = candidate('aspirin'), candidate('warfarin')
    combo = Candidate(tuple(sorted(aspirin.ingredient_ids + warfarin.ingredient_ids)),
                      aspirin.names + warfarin.names, .8, 'fixture')
    medication = Medication('unreadable', None, mention_id='m1', candidates=(combo, aspirin))
    assessment = candidate_assessments([medication])[0]
    assert assessment['status'] == 'conditional'
    assert assessment['stable_states'] == []
    assert assessment['possible_states'] == ['high', 'no_ingredient_pair']
