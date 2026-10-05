from streamlit.testing.v1 import AppTest
from pathlib import Path

APP = Path(__file__).resolve().parents[1] / 'llm.py'


def test_review_desk_builds_and_corrects_a_dossier():
    app = AppTest.from_file(APP, default_timeout=30).run()
    assert not app.exception
    app.text_area(key='med_input').set_value('Warfarln 5 mg\nAspirin 75 mg')
    next(b for b in app.button if b.label == 'Build review dossier').click().run()
    assert not app.exception
    assert app.session_state['result'].completeness == 'incomplete'
    app.selectbox(key='candidate_m1').select('1. warfarin').run()
    app.button(key='confirm_m1').click().run()
    assert not app.exception
    assert app.session_state['result'].alerts[0].severity == 'high'
    assert len(app.session_state['result'].review_history) == 1


def test_empty_input_and_navigation():
    app = AppTest.from_file(APP, default_timeout=30).run()
    next(b for b in app.button if b.label == 'Build review dossier').click().run()
    assert not app.exception
    assert app.session_state['result'].completeness == 'empty'
    app.radio[0].set_value('Methods').run()
    assert not app.exception


def test_typed_graph_exposes_candidate_paths_and_review_changes():
    app = AppTest.from_file(APP, default_timeout=30).run()
    app.text_area(key='med_input').set_value('Warfarln\nAspirin')
    next(b for b in app.button if b.label == 'Build review dossier').click().run()
    app.radio[0].set_value('Interaction map').run()
    assert not app.exception
    assert app.selectbox(key='graph_mention').value == 'm1'
    app.radio[0].set_value('Review desk').run()
    app.selectbox(key='candidate_m1').select('1. warfarin').run()
    app.button(key='confirm_m1').click().run()
    app.radio[0].set_value('Interaction map').run()
    assert not app.exception
    assert app.selectbox(key='graph_finding').value == 'DDInter1951|DDInter20'
    assert app.session_state['graph_comparison']['introduced'] == ['DDInter1951|DDInter20']


def test_changed_input_blocks_graph_export_until_rebuilt():
    app = AppTest.from_file(APP, default_timeout=30).run()
    app.text_area(key='med_input').set_value('Warfarin\nAspirin')
    next(b for b in app.button if b.label == 'Build review dossier').click().run()
    app.text_area(key='med_input').set_value('Warfarin').run()
    app.radio[0].set_value('Interaction map').run()
    assert not app.exception
    assert any('input has changed' in warning.value for warning in app.warning)
    assert not any(s.key == 'graph_focus' for s in app.selectbox)
