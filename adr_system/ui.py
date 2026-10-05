"""Streamlit presentation; all screening and research logic lives outside the UI."""
from __future__ import annotations

import hashlib
import io
import json
import tempfile
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st
from PIL import Image, ImageDraw, ImageOps

from .data import ROOT
from .engine import analyze_medications, analyze_structured, apply_review, severity_rank
from .evidence import enrich_live
from .evidence_graph import build_evidence_graph, compare_graphs
from .explanations import PROMPT_VERSION, bounded_llm_explanation, configured_model, explanation_payload
from .normalization import normalise_foods, parse_medication_lines
from .report import DISCLAIMER, render_html_report, render_json_report, replay_report

CASES = {
    'Choose an example': ('', ''),
    '01 / Known source interaction': ('Warfarin 5 mg\nAspirin 75 mg', ''),
    '02 / Ambiguous medicine reading': ('Warfarln 5 mg\nAspirin 75 mg\nMetformln 500 mg', ''),
    '03 / Food consumption needs context': ('Simvastatin 10 mg', 'no grapefruit'),
    '04 / Incomplete assessment': ('Abacavir\nUnidentified medicine', ''),
    '05 / Unknown source severity': ('Acetaminophen\nAspirin', ''),
    '06 / Biological pathway': ('Clarithromycin\nMidazolam', ''),
    '07 / Supplement pathway': ('Midazolam', "St. John's wort"),
}


def _load_case():
    meds, foods = CASES[st.session_state['case_choice']]
    st.session_state['med_input'], st.session_state['food_input'] = meds, foods
    for key in ('result', 'ocr_document', 'source_image', 'explanation_cache', 'graph_comparison'):
        st.session_state.pop(key, None)


def _set_result(result):
    previous = st.session_state.get('result')
    if previous is not None and previous.run_id == result.run_id:
        st.session_state['graph_comparison'] = compare_graphs(
            build_evidence_graph(previous, include_candidates=False),
            build_evidence_graph(result, include_candidates=False))
    else:
        st.session_state.pop('graph_comparison', None)
    st.session_state['result'] = result
    st.session_state.pop('explanation_cache', None)


def _input_signature():
    return (st.session_state['med_input'], st.session_state['food_input'],
            bool(st.session_state.get('enable_local_model', False)))


def _stale_result():
    return st.session_state.get('analysis_input', _input_signature()) != _input_signature()


def _ocr_upload(upload):
    from paddle_ocr import extract_document
    if upload.size > 10 * 1024 * 1024:
        raise ValueError('Use an image smaller than 10 MB.')
    content = upload.getvalue()
    with Image.open(io.BytesIO(content)) as image:
        if image.width * image.height > 20_000_000:
            raise ValueError('Use an image smaller than 20 megapixels.')
        source = ImageOps.exif_transpose(image).convert('RGB')
    with tempfile.TemporaryDirectory(prefix='medication-review-') as directory:
        path = Path(directory) / 'input.png'
        source.save(path)
        document = extract_document(str(path))
    if not document['lines']:
        raise ValueError('No text was detected. Crop the medication section or type the medicines.')
    st.session_state['source_image'] = source
    st.session_state['ocr_document'] = document
    st.session_state['med_input'] = document['text']
    st.session_state.pop('result', None)


def _header():
    st.markdown('''<div class="hero"><div class="eyebrow">MEDICATION REVIEW / EVIDENCE DOSSIER</div>
<h1>Read carefully.<br>Review what matters.</h1><p>Trace a medicine from prescription text to ingredient identity, interaction record, and evidence.</p></div>''', unsafe_allow_html=True)
    st.caption(DISCLAIMER)


def _input_panel():
    st.subheader('01 / Add a prescription')
    left, right = st.columns([3, 2], gap='large')
    with right:
        upload = st.file_uploader('Printed prescription image', type=['png', 'jpg', 'jpeg'],
            help='English printed text. Crop to medicines. Images stay in this session; OCR runs locally.')
        if st.button('Read prescription image', disabled=upload is None):
            try:
                with st.spinner('Reading text locally. The first run may need OCR model downloads…'):
                    _ocr_upload(upload)
            except Exception as error:
                st.error(f'OCR could not complete ({type(error).__name__}). Check the OCR setup or enter medicines manually.')
                st.caption(str(error)[:240] if isinstance(error, ValueError) else 'See the local setup and manual-testing guide.')
        if 'source_image' in st.session_state:
            st.image(st.session_state['source_image'], caption='Source image · session only', width='stretch')
        st.caption('External evidence and model calls are off by default. Clearing the session removes the loaded image and review history.')
    with left:
        st.selectbox('Synthetic examples', list(CASES), key='case_choice', on_change=_load_case)
        st.text_area('Medication text', key='med_input', height=180,
                     placeholder='Warfarin 5 mg\nAspirin 75 mg', max_chars=10000)
        st.text_area('Foods and drinks', key='food_input', height=80, max_chars=2000,
                     placeholder='Grapefruit juice; no spinach')
        research_model = st.checkbox('Use the local research model for documented interactions with unknown severity',
                                     value=False, key='enable_local_model')
        st.caption('Candidate screening and biological paths run locally. Model predictions remain separate from source findings.')
        if st.button('Build review dossier', type='primary'):
            try:
                with st.spinner('Resolving ingredients and checking source records…'):
                    document = st.session_state.get('ocr_document')
                    if document and document['text'] == st.session_state['med_input']:
                        from paddle_ocr import medications_from_document
                        result = analyze_structured(medications_from_document(document), normalise_foods(st.session_state['food_input']),
                                                    enable_research_model=research_model)
                        result.manifest = {**result.manifest, 'ocr': {k: v for k, v in document.items() if k not in ('text', 'lines')}}
                    else:
                        result = analyze_medications(st.session_state['med_input'], st.session_state['food_input'],
                                                     enable_research_model=research_model)
                    _set_result(result)
                    st.session_state['analysis_input'] = _input_signature()
            except (ValueError, OSError) as error:
                st.error(str(error))


def _review_panel(result):
    st.subheader('02 / Verify medicine identities')
    unresolved = sum(not m.normalized and m.status != 'excluded' for m in result.medications)
    a, b, c, d = st.columns(4)
    a.metric('Medication entries', len(result.medications))
    b.metric('Unresolved', unresolved)
    c.metric('Source findings', len(result.alerts))
    d.metric('Unassessed pairs', sum(p['state'] == 'unassessed' for p in result.pair_assessments))
    for warning in result.warnings:
        st.warning(warning)
    st.dataframe([{'Entry': m.mention_id, 'Observed text': m.name, 'Ingredient': ' + '.join(m.ingredient_names) or 'Review required',
                   'Status': m.status, 'Strength': m.strength or '', 'Frequency': m.frequency or '',
                   'Route': m.route or ''} for m in result.medications], hide_index=True, width='stretch')
    meds = {m.mention_id: m for m in result.medications}
    queue = result.review_queue
    if not queue:
        st.success('All entered items have a recorded review decision.')
        return
    st.caption('Review order reflects candidate-dependent source findings. It is not a patient risk score. Exact matches remain reviewable.')
    selected = st.selectbox('Next entry to review', [q.mention_id for q in queue],
                            format_func=lambda ident: f'{ident} · {meds[ident].name}', key='review_selection')
    med = meds[selected]
    item = next(q for q in queue if q.mention_id == selected)
    left, right = st.columns([3, 2], gap='large')
    with right:
        if med.bbox and 'source_image' in st.session_state:
            preview = st.session_state['source_image'].copy()
            ImageDraw.Draw(preview).rectangle(med.bbox, outline='#c04432', width=5)
            st.image(preview, caption='Highlighted source region', width='stretch')
        if med.ocr_score is not None:
            st.caption(f'OCR recognition score: {med.ocr_score:.3f}; not calibrated clinical confidence.')
    with left:
        st.info(item.reason)
        options = ['Choose a resolution'] + [f'{i + 1}. {" + ".join(c.names)}' for i, c in enumerate(med.candidates)]
        options += ['Enter ingredients manually', 'Exclude non-medication text']
        choice = st.selectbox('Resolution', options, key=f'candidate_{selected}')
        manual = st.text_input('Manual ingredient names, separated by commas', key=f'manual_{selected}') if choice == 'Enter ingredients manually' else ''
        if med.candidates:
            st.caption('Candidates are lexical suggestions. Confirm the identity from the prescription, not from the number of alerts.')
        if st.button('Confirm review decision', key=f'confirm_{selected}', type='primary'):
            try:
                action = 'resolve'
                if choice == 'Choose a resolution':
                    raise ValueError('Choose a candidate or enter the medicine manually.')
                if choice == 'Exclude non-medication text':
                    ids, action = (), 'exclude_non_medication'
                elif choice == 'Enter ingredients manually':
                    entries = parse_medication_lines(manual)
                    if not entries or any(not m.normalized for m in entries):
                        raise ValueError('Manual entries must resolve to known ingredient names; unresolved identities remain unassessed.')
                    ids = tuple(sorted({i for m in entries for i in m.ingredient_ids}))
                else:
                    ids = med.candidates[int(choice.split('.')[0]) - 1].ingredient_ids
                next_result = apply_review(result, selected, ids, action=action)
                if 'ocr' in result.manifest:
                    next_result.manifest = {**next_result.manifest, 'ocr': result.manifest['ocr']}
                _set_result(next_result)
                st.rerun()
            except ValueError as error:
                st.error(str(error))
    if item.previews:
        with st.expander('How candidate interpretations change source findings'):
            st.caption('Hypothetical candidate previews. These are not resolved findings or prescribing advice.')
            st.dataframe(item.previews, hide_index=True, width='stretch')


def _findings_panel(result):
    st.subheader('03 / Inspect findings and evidence')
    st.caption(f'Completeness: {result.completeness} · Run {result.run_id}')
    if not result.alerts:
        st.info('No known interaction found in this snapshot. Incomplete or missing coverage is not evidence of safety.')
    for alert in sorted(result.alerts, key=lambda a: severity_rank(a.severity), reverse=True):
        with st.container(border=True):
            st.markdown(f'**{alert.severity.upper()} · {" + ".join(alert.entities)}**')
            st.write(alert.mechanism)
            st.caption(f'{alert.interaction_type} · {alert.source_record_id} · Entries {", ".join(alert.mention_ids)}')
            for conflict in alert.source_disagreements:
                st.warning(conflict)
            with st.expander('Source records and passages'):
                for evidence in alert.evidence:
                    st.write(evidence.title)
                    from .report import safe_url
                    if safe_url(evidence.url) != '#':
                        st.link_button('Open source', evidence.url)
                    st.caption(f'{evidence.purpose} · {evidence.evidence_id} · {evidence.document_version}')
                    st.write(evidence.excerpt)
    if st.button('Fetch / refresh supporting evidence', help='Sends normalized ingredient names to openFDA and PubChem. Raw input and images are not sent.'):
        before = build_evidence_graph(result, include_candidates=False)
        with st.spinner('Retrieving relevant source passages…'):
            enrich_live(result, refresh=True)
            st.session_state['graph_comparison'] = compare_graphs(before, build_evidence_graph(result, include_candidates=False))
            st.session_state.pop('explanation_cache', None)
        st.rerun()
    external = st.checkbox('Allow Groq to select source excerpts from normalized findings', key='allow_llm',
                           help='Optional external request. Original prescription text and images are excluded.')
    key = hashlib.sha256(json.dumps(explanation_payload(result), sort_keys=True).encode()).hexdigest() + str(external) + configured_model() + PROMPT_VERSION
    cached = st.session_state.get('explanation_cache')
    if not cached or cached[0] != key:
        with st.spinner('Preparing explanation…'):
            explanation, mode = bounded_llm_explanation(result, enabled=external)
        st.session_state['explanation_cache'] = (key, explanation, mode)
    _, explanation, mode = st.session_state['explanation_cache']
    st.caption(mode)
    st.write(explanation)
    with st.expander('Coverage and source status'):
        st.json({'pairs': result.pair_assessments, 'foods': result.food_assessments, 'sources': result.source_status})
    return explanation


def _export_panel(result, explanation):
    st.subheader('04 / Export and reproduce')
    original = st.checkbox('Include original entered text in downloads', value=False)
    st.caption('Default downloads contain normalized identities and unresolved entry IDs. Medication information can still be sensitive.')
    a, b = st.columns(2)
    a.download_button('Download evidence report', render_html_report(result, explanation, include_original=original),
                      file_name=f'dossier-{result.run_id}.html', mime='text/html')
    b.download_button('Download replayable JSON', render_json_report(result, include_original=original),
                      file_name=f'dossier-{result.run_id}.json', mime='application/json')
    with st.expander('Review history and reproducibility manifest'):
        st.json({'history': [d.__dict__ for d in result.review_history], 'manifest': result.manifest})


def _hybrid_panel(result):
    st.subheader('Automatic screening and research insights')
    candidate_tab, mechanism_tab, model_tab = st.tabs(['Uncertain identities', 'Biological paths', 'Model predictions'])
    with candidate_tab:
        items = [a for a in result.candidate_assessments if a['status'] != 'resolved']
        st.caption('Stable conclusions hold across retained candidates only. The correct medicine may be missing from that set.')
        if items:
            st.dataframe([{'Entries': ' + '.join(a['mention_ids']), 'Status': a['status'],
                           'Stable source states': ', '.join(a['stable_states']),
                           'Possible source states': ', '.join(a['possible_states']),
                           'Candidate combinations': len(a['outcomes'])} for a in items], hide_index=True, width='stretch')
            with st.expander('Candidate identities and supporting record IDs'):
                st.json(items)
        else:
            st.info('No unresolved candidate pairs remain.')
    with mechanism_tab:
        st.caption('Sourced enzyme and transporter paths suggest possibilities; they do not establish interaction severity.')
        if result.mechanism_hypotheses:
            for item in result.mechanism_hypotheses:
                with st.expander(' + '.join(item.get('entities', [])) or 'Biological pathway'):
                    st.json(item)
        else:
            st.info('No compatible biological paths were found in this bounded catalog.')
    with model_tab:
        status = result.hybrid_status.get('model', {})
        st.caption('Predictions estimate source labels for documented interactions. Scores are not probabilities of patient harm.')
        if result.model_predictions:
            st.dataframe([{'Ingredients': ' + '.join(p.get('ingredient_ids', [])),
                           'Predicted source severity': (p.get('predicted_severity') or 'Unavailable')
                                                       if p.get('accepted') else 'Abstained',
                           'Accepted': p.get('accepted', False),
                           'Reason': p.get('abstention_reason', ''),
                           'Model': p.get('model_version', '')} for p in result.model_predictions],
                         hide_index=True, width='stretch')
            with st.expander('Prediction scores and model rationale'):
                st.json(result.model_predictions)
        else:
            st.info('No experimental predictions are available. Enable the local model and rebuild to assess eligible source-unknown pairs.')
        with st.expander('Local component availability'):
            st.json(result.hybrid_status)


def _interaction_map(result):
    import networkx as nx
    if not result.alerts:
        st.info('The map appears when source findings are available.')
        return
    graph = nx.Graph()
    for alert in result.alerts:
        graph.add_edge(*alert.entities)
    positions = nx.spring_layout(graph, seed=7)
    colors = {'high': '#b43e2d', 'moderate': '#bb7d16', 'low': '#24756c', 'unknown': '#657782'}
    fig = go.Figure()
    for alert in result.alerts:
        first, second = alert.entities
        fig.add_trace(go.Scatter(x=[positions[first][0], positions[second][0]], y=[positions[first][1], positions[second][1]],
            mode='lines', line=dict(color=colors[alert.severity], width=3),
            text=f'{alert.severity} | {alert.source_record_id}', hoverinfo='text', showlegend=False))
    nodes = sorted(graph.nodes)
    fig.add_trace(go.Scatter(x=[positions[n][0] for n in nodes], y=[positions[n][1] for n in nodes], mode='markers+text',
        text=nodes, textposition='top center', marker=dict(size=20, color='#163334'), showlegend=False))
    fig.update_layout(height=500, xaxis=dict(visible=False), yaxis=dict(visible=False), margin=dict(l=25, r=25, t=30, b=25))
    st.plotly_chart(fig, width='stretch')
    record = st.selectbox('Inspect an edge record', [a.source_record_id for a in result.alerts])
    alert = next(a for a in result.alerts if a.source_record_id == record)
    st.json(alert.to_dict())


def evidence_graph_panel(result):
    import networkx as nx
    from html import escape
    dossier_graph = build_evidence_graph(result)
    graph = dossier_graph.graph
    summary = dossier_graph.summary()
    st.subheader('Follow the evidence')
    st.caption('Trace readings, ingredient identities, source findings, supporting passages and review decisions. Dashed relationships show candidate-dependent possibilities.')
    a, b, c, d = st.columns(4)
    a.metric('Entities', summary['nodes'])
    b.metric('Relationships', summary['edges'])
    c.metric('Detected findings', len(dossier_graph.nodes_of_kind('finding')))
    d.metric('Review decisions', len(result.review_history))
    if summary['candidate_previews_truncated']:
        st.warning(f"Displaying {summary['candidate_previews_included']} of {summary['candidate_previews_total']} candidate previews. Detected findings and their evidence remain included.")
    focus_order = {'finding': 0, 'mention': 1, 'review_decision': 2, 'source_record': 3,
                   'evidence': 4, 'ingredient': 5, 'candidate': 6}
    choices = sorted(graph, key=lambda n: (focus_order.get(graph.nodes[n]['kind'], 9), graph.nodes[n]['label'], n))
    def entity_label(ident):
        data = graph.nodes[ident]
        detail = data.get('evidence_id', '')[:8]
        if data['kind'] == 'source_document':
            detail = f"{data['document_id'] or 'no specific ID'} · version {data['document_version'] or 'unavailable'}"
        elif data['kind'] == 'candidate':
            detail = f"entry {data['mention_id']}"
        return f"{data['kind'].replace('_', ' ').title()} · {data['label']}" + (f' · {detail}' if detail else '')
    focus = st.selectbox('Entity to explore', choices, key='graph_focus',
                         format_func=entity_label)
    hops = st.slider('Relationship depth', 1, 3, 2, key='graph_depth')
    view, clipped = dossier_graph.neighborhood(focus, hops=hops)
    if clipped:
        st.caption('The view is limited to 100 entities. Choose a closer focus or reduce the relationship depth to inspect other parts.')
    positions = nx.spring_layout(view.to_undirected(), seed=17)
    colors = {'mention': '#c04432', 'ingredient': '#24756c', 'finding': '#173d39',
              'evidence': '#396d98', 'source_document': '#62758b', 'source_record': '#445b55',
              'candidate': '#b98124', 'candidate_preview': '#b98124', 'review_priority': '#b98124',
              'review_decision': '#c04432', 'source_disagreement': '#c04432'}
    fig = go.Figure()
    for first, second, attrs in view.edges(data=True):
        hypothetical = attrs['relation'] in {'has_candidate', 'contains_ingredient', 'has_preview',
                                             'assumes_candidate', 'assumes_partner', 'could_retrieve'}
        fig.add_trace(go.Scatter(x=[positions[first][0], positions[second][0]],
            y=[positions[first][1], positions[second][1]], mode='lines',
            line=dict(color='#b98124' if hypothetical else '#a1b5ae', width=1.5,
                      dash='dot' if hypothetical else 'solid'),
            text=f"{escape(graph.nodes[first]['label'])} → {attrs['relation'].replace('_', ' ')} → {escape(graph.nodes[second]['label'])}",
            hoverinfo='text', showlegend=False))
    for kind in sorted({view.nodes[n]['kind'] for n in view}):
        nodes = [n for n in sorted(view) if view.nodes[n]['kind'] == kind]
        fig.add_trace(go.Scatter(x=[positions[n][0] for n in nodes], y=[positions[n][1] for n in nodes],
            mode='markers', name=kind.replace('_', ' ').title(),
            text=[escape(view.nodes[n]['label']) for n in nodes], hoverinfo='text',
            marker=dict(size=[22 if n == focus else 13 for n in nodes], color=colors.get(kind, '#81918a'),
                        line=dict(color='white', width=1))))
    fig.update_layout(height=540, xaxis=dict(visible=False), yaxis=dict(visible=False),
                      legend=dict(orientation='h', y=-.05), margin=dict(l=10, r=10, t=15, b=60))
    st.plotly_chart(fig, width='stretch', key='typed_graph_plot')
    with st.expander('Selected entity and its typed relationships'):
        st.json(dict(graph.nodes[focus], node_id=focus))
        rows = [{'From': graph.nodes[first]['label'], 'Relationship': attrs['relation'],
                 'To': graph.nodes[second]['label']}
                for first, second, attrs in graph.edges(data=True) if focus in (first, second)]
        st.dataframe(rows, hide_index=True, width='stretch')
    evidence, review = st.tabs(['Finding provenance', 'Identity review and changes'])
    with evidence:
        records = sorted({a.source_record_id for a in result.alerts})
        if records:
            selected = st.selectbox('Finding to trace', records, key='graph_finding')
            trace = dossier_graph.trace_finding(selected)
            rows = []
            for path in trace['support_paths']:
                mention, ingredient, finding, passage, document = [graph.nodes[n] for n in path]
                rows.append({'Entry': mention['mention_id'], 'Ingredient': ingredient['label'],
                             'Finding': finding['source_record_id'], 'Evidence ID': passage['evidence_id'],
                             'Source': document['source'], 'Document': document['document_id'],
                             'Version': document['document_version']})
            st.dataframe(rows, hide_index=True, width='stretch')
            st.caption('Paths document where a finding came from. Expert review is still needed to assess passage relevance.')
            for passage in trace['evidence']:
                with st.expander(f"{passage['label']} · {passage['evidence_id']}"):
                    st.write(passage['excerpt'])
                    st.caption(passage['purpose'])
            for disagreement in trace['source_disagreements']:
                st.warning(disagreement)
        else:
            st.info('There are no detected source findings. Unresolved entries can still have candidate-dependent possibilities in the review tab.')
    with review:
        if result.medications:
            selected = st.selectbox('Entry to inspect', [m.mention_id for m in result.medications], key='graph_mention')
            impacts = dossier_graph.candidate_impacts(selected)
            st.caption(impacts['scope'])
            if not impacts['complete']:
                st.warning('The exported preview set is bounded; this view does not enumerate every possible candidate combination.')
            rows = [{'Candidate': ' + '.join(p['candidate_names']), 'Partner entry': p['partner_mention'],
                     'Partner interpretation': ' + '.join(p['partner_names']),
                     'Source states': ', '.join(p['states']),
                     'Possible source records': ', '.join(r['source_record_id'] for r in p['records'])}
                    for p in impacts['previews']]
            st.dataframe(rows, hide_index=True, width='stretch')
            decisions = dossier_graph.review_changes(selected)
            if decisions:
                st.json(decisions)
            else:
                st.caption('No review decision has been recorded for this entry.')
        if st.session_state.get('graph_comparison') is not None:
            st.write('Changes after the latest correction or evidence refresh')
            st.json(st.session_state['graph_comparison'])
    original = st.checkbox('Include original prescription text in graph download', value=False, key='graph_original')
    exported = build_evidence_graph(result, include_original=True) if original else dossier_graph
    st.download_button('Download typed evidence graph', json.dumps(exported.to_dict(), indent=2, ensure_ascii=False),
                       file_name=f'evidence-graph-{result.run_id}.json', mime='application/json')
    with st.expander('Graph schema and coverage'):
        st.json(summary)


def network_graph(result):
    evidence, interactions = st.tabs(['Evidence graph', 'Interaction map'])
    with evidence:
        evidence_graph_panel(result)
    with interactions:
        _interaction_map(result)


def research_page():
    st.subheader('Research workspace')
    st.write('Versioned experiments generate case-level outputs, comparison tables, uncertainty estimates, and figures. Synthetic checks are software experiments, not clinical validation.')
    st.code('python scripts/evaluate.py --output artifacts/evaluation\npython scripts/benchmark.py --output artifacts/research\npython scripts/ocr_smoke.py')
    path = ROOT / 'artifacts' / 'research' / 'summary.json'
    if path.exists():
        st.json(json.loads(path.read_text(encoding='utf-8')))
        for chart in sorted(path.parent.glob('*.png')):
            st.image(str(chart), caption=chart.stem.replace('_', ' '))
    st.caption('Annotation instructions and manual acceptance steps are in docs/MANUAL_VALIDATION.md and docs/RESEARCH_PROTOCOL.md.')
    with st.expander('Hybrid engine experiments'):
        st.code('python scripts/hybrid_benchmark.py\npython scripts/hybrid_ocr_benchmark.py\n'
                'python scripts/train_ml.py --help\npython scripts/explain_ml.py --help\n'
                'python scripts/hybrid_model_robustness.py\npython scripts/verify_hybrid_release.py')
        experiments = [('Prescription text', ROOT / 'artifacts/hybrid/text'),
                       ('Prescription images', ROOT / 'artifacts/hybrid/ocr'),
                       ('Source-severity model comparison', ROOT / 'artifacts/ml/training-minibatch'),
                       ('Model recognition robustness', ROOT / 'artifacts/hybrid/model-recognition')]
        for title, folder in experiments:
            summary = folder / 'summary.json'
            if summary.exists():
                st.write(title)
                st.json(json.loads(summary.read_text(encoding='utf-8')))
                for chart in sorted(folder.glob('*.png')):
                    st.image(str(chart), caption=chart.stem.replace('_', ' '))
    uploaded = st.file_uploader('Replay a JSON dossier from this software/data version', type=['json'])
    if st.button('Verify and replay', disabled=uploaded is None):
        try:
            _set_result(replay_report(uploaded.getvalue().decode('utf-8')))
            st.session_state.pop('analysis_input', None)
            st.success('Local findings reproduced. Open the Review desk to inspect them.')
        except (ValueError, KeyError, TypeError) as error:
            st.error(f'Replay rejected: {error}')


def main():
    # Detach input values from widget cleanup when a different workspace is shown.
    st.session_state['med_input'] = st.session_state.get('med_input', '')
    st.session_state['food_input'] = st.session_state.get('food_input', '')
    st.session_state['enable_local_model'] = st.session_state.get('enable_local_model', False)
    st.sidebar.markdown('### Evidence dossier')
    page = st.sidebar.radio('Workspace', ['Review desk', 'Interaction map', 'Research workspace', 'Methods'])
    if st.sidebar.button('Clear session', type='secondary'):
        for key in list(st.session_state):
            del st.session_state[key]
        st.rerun()
    st.sidebar.caption('Local screening · source-linked outputs\n\nSource labels describe known records. They do not estimate individual risk.')
    _header()
    if page == 'Review desk':
        _input_panel()
        result = st.session_state.get('result')
        if result is not None:
            if _stale_result():
                st.warning('The input has changed, or the model setting has changed. Build a new review dossier before reviewing or exporting these edits.')
                return
            st.divider()
            _review_panel(result)
            explanation = _findings_panel(result)
            _hybrid_panel(result)
            _export_panel(result, explanation)
    elif page == 'Interaction map':
        if 'result' in st.session_state:
            if _stale_result():
                st.warning('The input has changed, or the model setting has changed. Build a new review dossier before exploring or exporting its graph.')
            else:
                network_graph(st.session_state['result'])
        else:
            st.info('Build a dossier first.')
    elif page == 'Research workspace':
        research_page()
    else:
        st.subheader('Methods and boundaries')
        st.write('Prescription recognition → candidate-set screening → documented source lookup → biological pathways and experimental model estimates → source-linked report. Candidate ranking is based on lexical similarity; interaction impact only orders optional identity review.')
        st.write('Candidate-set joins screen plausible medicine identities automatically. A sourced biological role graph supplies separate enzyme/transporter possibilities. An optional locally trained molecular/GraphSAGE model estimates source severity for documented interactions whose severity is unknown.')
        st.write('Documented source findings, biological possibilities and model predictions have separate types and provenance. Model rationale describes influence on a prediction; it does not establish a biological cause. Statistical scores do not estimate patient harm.')
        st.write('Resolved identities yield documented findings. Unresolved entries receive candidate-dependent assessments within retained alternatives. Missing records and unknown severity never imply safety. Dose, organ function, pregnancy, and medical history are not used to predict risk.')
        st.write('Core screening runs offline. Optional evidence sends normalized ingredient names to public sources. Optional Groq requests contain allowlisted source records only. Images and original text stay in session unless you explicitly include text in a downloaded report.')
        st.write('Automated experiments measure source-finding recovery under recognition errors, source-severity prediction, calibration, missing features and unseen-drug performance. Generated prescriptions provide controlled labels; clinical effectiveness has not been measured.')
