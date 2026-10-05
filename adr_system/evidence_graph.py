"""Typed dossier provenance and candidate-impact graph; no learned clinical inference."""
from __future__ import annotations

import hashlib
import itertools
import json
from collections import Counter, deque
from dataclasses import dataclass

import networkx as nx

from .data import load_food_rules
from .knowledge import pair_alert
from .models import AnalysisResult, Evidence, InteractionAlert
from .terminology import catalog

GRAPH_SCHEMA_VERSION = '1.1'
DEFAULT_PREVIEW_LIMIT = 120
MAX_PREVIEW_PAIRS = 32
NODE_KINDS = {'dossier', 'mention', 'ingredient', 'candidate', 'finding', 'source_record',
              'evidence', 'source_document', 'review_decision', 'historical_finding',
              'review_priority', 'candidate_preview', 'coverage', 'food', 'source_disagreement',
              'candidate_assessment', 'mechanism_hypothesis', 'model_prediction', 'model_rationale',
              'biological_role', 'biological_target'}
RELATIONS = {
    'has_mention': ({'dossier'}, {'mention'}),
    'resolved_to': ({'mention'}, {'ingredient'}),
    'has_candidate': ({'mention'}, {'candidate'}),
    'contains_ingredient': ({'candidate'}, {'ingredient'}),
    'has_finding': ({'dossier'}, {'finding'}),
    'participates_in': ({'ingredient', 'food'}, {'finding'}),
    'uses_record': ({'finding'}, {'source_record'}),
    'record_involves': ({'source_record'}, {'ingredient', 'food'}),
    'supported_by': ({'finding'}, {'evidence'}),
    'identity_context': ({'finding'}, {'evidence'}),
    'has_evidence': ({'source_record'}, {'evidence'}),
    'from_document': ({'evidence'}, {'source_document'}),
    'has_disagreement': ({'finding', 'source_record'}, {'source_disagreement'}),
    'has_review': ({'dossier'}, {'review_decision'}),
    'changes_mention': ({'review_decision'}, {'mention'}),
    'selected_ingredient': ({'review_decision'}, {'ingredient'}),
    'previous_ingredient': ({'review_decision'}, {'ingredient'}),
    'introduced': ({'review_decision'}, {'historical_finding'}),
    'removed': ({'review_decision'}, {'historical_finding'}),
    'retained': ({'review_decision'}, {'historical_finding'}),
    'currently_detected_as': ({'historical_finding'}, {'finding'}),
    'has_priority': ({'mention'}, {'review_priority'}),
    'has_preview': ({'review_priority'}, {'candidate_preview'}),
    'assumes_candidate': ({'candidate_preview'}, {'candidate'}),
    'assumes_partner': ({'candidate_preview'}, {'candidate', 'mention'}),
    'could_retrieve': ({'candidate_preview'}, {'source_record'}),
    'has_coverage': ({'dossier'}, {'coverage'}),
    'assesses': ({'coverage'}, {'mention', 'ingredient', 'food'}),
    'has_candidate_assessment': ({'dossier'}, {'candidate_assessment'}),
    'assesses_identity': ({'candidate_assessment'}, {'mention'}),
    'has_mechanism_hypothesis': ({'dossier'}, {'mechanism_hypothesis'}),
    'hypothesis_involves': ({'mechanism_hypothesis'}, {'ingredient'}),
    'has_biological_role': ({'mechanism_hypothesis'}, {'biological_role'}),
    'plays_role': ({'ingredient', 'food'}, {'biological_role'}),
    'role_targets': ({'biological_role'}, {'biological_target'}),
    'role_source': ({'biological_role'}, {'source_document'}),
    'has_model_prediction': ({'dossier'}, {'model_prediction'}),
    'prediction_involves': ({'model_prediction'}, {'ingredient'}),
    'has_model_rationale': ({'model_prediction'}, {'model_rationale'}),
}


def _id(kind: str, *parts) -> str:
    value = json.dumps(parts, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
    return f'{kind}:{hashlib.sha256(value.encode()).hexdigest()[:24]}'


@dataclass
class EvidenceGraph:
    graph: nx.MultiDiGraph

    def nodes_of_kind(self, kind: str) -> list[str]:
        return sorted(n for n, data in self.graph.nodes(data=True) if data['kind'] == kind)

    def _targets(self, node: str, relation: str) -> list[str]:
        return sorted({target for _, target, data in self.graph.out_edges(node, data=True)
                       if data['relation'] == relation})

    def finding_nodes(self, record_id: str) -> list[str]:
        return [n for n in self.nodes_of_kind('finding') if self.graph.nodes[n]['source_record_id'] == record_id]

    def evidence_for(self, record_id: str, *, include_identity: bool = False) -> list[dict]:
        """Retrieve only passages connected to actual findings by allowed relations."""
        items = {}
        for finding in self.finding_nodes(record_id):
            for _, target, data in self.graph.out_edges(finding, data=True):
                if data['relation'] == 'supported_by' or (include_identity and data['relation'] == 'identity_context'):
                    item = self.graph.nodes[target]
                    items[target] = (data.get('ordinal', 0), dict(item, node_id=target))
        return [item for _, item in sorted(items.values(), key=lambda x: (x[0], x[1]['node_id']))]

    def trace_finding(self, record_id: str) -> dict:
        """Paths are mention -> resolved ingredient -> finding -> passage -> document.

        Candidate and identity-only edges are never treated as support paths.
        """
        paths, findings, disagreements = [], self.finding_nodes(record_id), []
        for finding in findings:
            disagreements.extend(self.graph.nodes[n]['text'] for n in self._targets(finding, 'has_disagreement'))
            for evidence in self._targets(finding, 'supported_by'):
                documents = self._targets(evidence, 'from_document')
                for ingredient, _, edge in self.graph.in_edges(finding, data=True):
                    if edge['relation'] != 'participates_in' or self.graph.nodes[ingredient]['kind'] != 'ingredient':
                        continue
                    for mention, _, identity in self.graph.in_edges(ingredient, data=True):
                        if identity['relation'] == 'resolved_to':
                            for document in documents:
                                paths.append([mention, ingredient, finding, evidence, document])
        return {'record_id': record_id, 'finding_nodes': findings,
                'support_paths': sorted(paths), 'evidence': self.evidence_for(record_id),
                'source_disagreements': sorted(set(disagreements)),
                'scope': 'Recorded provenance paths; clinical relevance still requires review.'}

    def candidate_impacts(self, mention_id: str) -> dict:
        mentions = [n for n in self.nodes_of_kind('mention') if self.graph.nodes[n]['mention_id'] == mention_id]
        if not mentions:
            raise ValueError('Unknown medication mention.')
        previews = []
        for mention in mentions:
            for priority in self._targets(mention, 'has_priority'):
                for preview in self._targets(priority, 'has_preview'):
                    item = dict(self.graph.nodes[preview], node_id=preview)
                    item['records'] = [dict(self.graph.nodes[n], node_id=n)
                                       for n in self._targets(preview, 'could_retrieve')]
                    previews.append(item)
        return {'mention_id': mention_id, 'previews': sorted(previews, key=lambda p: p['ordinal']),
                'complete': self.graph.graph['candidates_included'] and not self.graph.graph['candidate_previews_truncated'] and
                            not any(p['pair_checks_truncated'] for p in previews),
                'scope': 'Candidate-dependent possibilities; not detected findings.'}

    def unresolved_for_record(self, record_id: str) -> list[str]:
        return sorted({self.graph.nodes[n]['mention_id'] for n in self.nodes_of_kind('mention')
                       if not self.graph.nodes[n]['normalized'] and self.graph.nodes[n]['status'] != 'excluded'
                       and any(r['source_record_id'] == record_id
                               for p in self.candidate_impacts(self.graph.nodes[n]['mention_id'])['previews']
                               for r in p['records'])})

    def review_changes(self, mention_id: str) -> list[dict]:
        items = []
        for node in self.nodes_of_kind('review_decision'):
            data = self.graph.nodes[node]
            if data['mention_id'] == mention_id:
                changes = {rel: sorted(self.graph.nodes[n]['source_record_id'] for n in self._targets(node, rel))
                           for rel in ('introduced', 'removed', 'retained')}
                items.append(dict(data, node_id=node, **changes))
        return sorted(items, key=lambda item: item['ordinal'])

    def neighborhood(self, focus: str, *, hops: int = 2, limit: int = 100) -> tuple[nx.MultiDiGraph, bool]:
        if focus not in self.graph or not 0 <= hops <= 4 or not 1 <= limit <= 250:
            raise ValueError('Invalid graph focus, depth or display limit.')
        seen, pending, truncated = {focus}, deque([(focus, 0)]), False
        while pending:
            node, depth = pending.popleft()
            if depth >= hops:
                continue
            for neighbor in sorted(set(self.graph.successors(node)) | set(self.graph.predecessors(node))):
                if neighbor in seen:
                    continue
                if len(seen) >= limit:
                    truncated = True
                    continue
                seen.add(neighbor)
                pending.append((neighbor, depth + 1))
        return self.graph.subgraph(sorted(seen)).copy(), truncated

    def to_dict(self) -> dict:
        data = {'schema_version': GRAPH_SCHEMA_VERSION, 'metadata': dict(self.graph.graph),
                'nodes': [dict(self.graph.nodes[n], id=n) for n in sorted(self.graph)],
                'edges': sorted([dict(attrs, source=a, target=b, id=key)
                                 for a, b, key, attrs in self.graph.edges(keys=True, data=True)], key=lambda e: e['id'])}
        raw = json.dumps(data, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
        data['content_sha256'] = hashlib.sha256(raw.encode()).hexdigest()
        return data

    def summary(self) -> dict:
        return {'schema_version': GRAPH_SCHEMA_VERSION, 'nodes': self.graph.number_of_nodes(),
                'edges': self.graph.number_of_edges(),
                'node_types': dict(sorted(Counter(d['kind'] for _, d in self.graph.nodes(data=True)).items())),
                'relation_types': dict(sorted(Counter(d['relation'] for _, _, d in self.graph.edges(data=True)).items())),
                'content_sha256': self.to_dict()['content_sha256'], **self.graph.graph}

    def validate(self):
        for _, node in self.graph.nodes(data=True):
            if node['kind'] not in NODE_KINDS:
                raise ValueError('Unknown graph node type.')
            if node['kind'] == 'evidence' and hashlib.sha256(node['excerpt'].encode()).hexdigest() != node['excerpt_hash']:
                raise ValueError('Graph evidence hash differs from the passage.')
        for first, second, edge in self.graph.edges(data=True):
            kinds = RELATIONS.get(edge['relation'])
            if not kinds or self.graph.nodes[first]['kind'] not in kinds[0] or self.graph.nodes[second]['kind'] not in kinds[1]:
                raise ValueError('Invalid typed graph relationship.')
            if edge['relation'] == 'supported_by' and self.graph.nodes[second]['purpose'] == 'identity_only':
                raise ValueError('Identity information cannot support an interaction finding.')


def build_evidence_graph(result: AnalysisResult, *, include_original: bool = False,
                         include_candidates: bool = True, preview_limit: int = DEFAULT_PREVIEW_LIMIT) -> EvidenceGraph:
    if not 0 <= preview_limit <= 500:
        raise ValueError('Use a preview limit between 0 and 500.')
    graph = nx.MultiDiGraph()
    graph.graph.update(run_id=result.run_id, completeness=result.completeness,
                       original_input_included=include_original, candidates_included=include_candidates,
                       candidate_preview_limit=preview_limit, candidate_previews_total=0,
                       candidate_previews_included=0, candidate_previews_truncated=False,
                       scope='Source provenance, candidate possibilities, and separate biological/model hypotheses.')
    names = catalog()[1]
    ids_by_name = {name: ident for ident, name in names.items()}
    medications_by_id = {m.mention_id: m for m in result.medications}

    def node(kind, parts, label, **attrs):
        ident = _id(kind, *parts)
        graph.add_node(ident, kind=kind, label=label, **attrs)
        return ident

    def edge(first, relation, second, **attrs):
        graph.add_edge(first, second, key=_id('edge', first, relation, second), relation=relation, **attrs)

    def ingredient(ident):
        return node('ingredient', (ident,), names[ident], ingredient_id=ident)

    def passage(item: Evidence):
        document = node('source_document', (item.source, item.document_id, item.document_version, item.url),
                        item.title, source=item.source, document_id=item.document_id,
                        document_version=item.document_version, url=item.url,
                        identifier_available=bool(item.document_id))
        evidence = node('evidence', (item.evidence_id, item.document_version), item.title,
                        evidence_id=item.evidence_id, excerpt=item.excerpt, excerpt_hash=item.excerpt_hash,
                        purpose=item.purpose, section=item.section, retrieved_at=item.retrieved_at)
        edge(evidence, 'from_document', document)
        return evidence

    def source_record(alert: InteractionAlert):
        record = node('source_record', (alert.source_record_id, alert.interaction_type, alert.data_version),
                      alert.source_record_id, source_record_id=alert.source_record_id,
                      interaction_type=alert.interaction_type, data_version=alert.data_version,
                      severity=alert.severity, ingredient_ids=list(alert.ingredient_ids))
        for ident in alert.ingredient_ids:
            edge(record, 'record_involves', ingredient(ident))
        for item in alert.evidence:
            if item.purpose != 'identity_only':
                edge(record, 'has_evidence', passage(item))
        for text in alert.source_disagreements:
            disagreement = node('source_disagreement', (record, text), 'Source disagreement', text=text)
            edge(record, 'has_disagreement', disagreement)
        return record

    dossier = node('dossier', (result.run_id,), 'Medication review dossier', completeness=result.completeness)
    mentions, candidates = {}, {}
    for med in result.medications:
        attrs = dict(mention_id=med.mention_id, status=med.status, normalized=med.normalized,
                     strength=med.strength, route=med.route, frequency=med.frequency,
                     source_span=med.source_span, bbox=med.bbox, ocr_score=med.ocr_score)
        if include_original:
            attrs.update(observed_text=med.name, unparsed_text=med.unparsed_text)
        mention = node('mention', (result.run_id, med.mention_id), f'Entry {med.mention_id}', **attrs)
        mentions[med.mention_id] = mention
        edge(dossier, 'has_mention', mention)
        for ident in med.ingredient_ids:
            edge(mention, 'resolved_to', ingredient(ident), resolution_status=med.status)
        if include_candidates:
            for choice in med.candidates:
                candidate = node('candidate', (mention, choice.ingredient_ids), ' + '.join(choice.names),
                                 mention_id=med.mention_id, ingredient_ids=list(choice.ingredient_ids), lexical_score=choice.score,
                                 method=choice.method, assertion='candidate; not a resolved identity')
                candidates[(med.mention_id, tuple(choice.names))] = candidate
                edge(mention, 'has_candidate', candidate)
                for ident in choice.ingredient_ids:
                    edge(candidate, 'contains_ingredient', ingredient(ident))

    findings = {}
    for alert in result.alerts:
        finding = node('finding', (result.run_id, alert.source_record_id, alert.interaction_type),
                       ' + '.join(alert.entities), source_record_id=alert.source_record_id,
                       severity=alert.severity, interaction_type=alert.interaction_type,
                       ingredient_ids=list(alert.ingredient_ids), data_version=alert.data_version,
                       assertion='detected source finding; not patient-specific risk')
        findings[alert.source_record_id] = finding
        edge(dossier, 'has_finding', finding)
        record = source_record(alert)
        edge(finding, 'uses_record', record)
        for ident in alert.ingredient_ids:
            edge(ingredient(ident), 'participates_in', finding)
        if alert.interaction_type == 'drug-food':
            food = node('food', (alert.source_record_id, alert.entities[-1]), alert.entities[-1], rule_id=alert.source_record_id)
            edge(food, 'participates_in', finding)
            edge(record, 'record_involves', food)
        for ordinal, item in enumerate(alert.evidence):
            edge(finding, 'identity_context' if item.purpose == 'identity_only' else 'supported_by',
                 passage(item), ordinal=ordinal)
        for text in alert.source_disagreements:
            conflict = node('source_disagreement', (record, text), 'Source disagreement', text=text)
            edge(finding, 'has_disagreement', conflict)

    for ordinal, decision in enumerate(result.review_history):
        review = node('review_decision', (result.run_id, ordinal), f'Review {ordinal + 1}', ordinal=ordinal,
                      mention_id=decision.mention_id, decided_at=decision.decided_at, action=decision.action)
        edge(dossier, 'has_review', review)
        edge(review, 'changes_mention', mentions[decision.mention_id])
        for ident in decision.ingredient_ids:
            edge(review, 'selected_ingredient', ingredient(ident))
        for ident in decision.previous_ids:
            edge(review, 'previous_ingredient', ingredient(ident))
        before, after = set(decision.before_findings), set(decision.after_findings)
        for relation, records in [('introduced', after - before), ('removed', before - after), ('retained', before & after)]:
            for record_id in sorted(records):
                historic = node('historical_finding', (result.run_id, record_id), record_id,
                                source_record_id=record_id, assertion='record ID captured in review history')
                edge(review, relation, historic)
                if record_id in findings:
                    edge(historic, 'currently_detected_as', findings[record_id])

    for ordinal, item in enumerate(result.pair_assessments):
        coverage = node('coverage', (result.run_id, 'pair', ordinal), item['state'], state=item['state'], assessment_type='pair')
        edge(dossier, 'has_coverage', coverage)
        for ident in item.get('ingredient_ids', []):
            edge(coverage, 'assesses', ingredient(ident))
        for mention_id in item.get('mention_ids', []):
            edge(coverage, 'assesses', mentions[mention_id])
    rules = {r['id']: r for r in load_food_rules()[0]['rules']}
    for ordinal, item in enumerate(result.food_assessments):
        rule = rules.get(item.get('rule_id'))
        label = rule['food_terms'][0] if rule else '[unmatched food entry]'
        food = node('food', (result.run_id, 'assertion', ordinal), label, rule_id=item.get('rule_id'),
                    **({'observed_text': item['food']} if include_original else {}))
        coverage = node('coverage', (result.run_id, 'food', ordinal), item['state'], state=item['state'], assessment_type='food')
        edge(dossier, 'has_coverage', coverage)
        edge(coverage, 'assesses', food)

    if include_candidates:
        graph.graph['candidate_previews_total'] = sum(len(q.previews) for q in result.review_queue)
        used = 0
        for rank, item in enumerate(result.review_queue):
            priority = node('review_priority', (result.run_id, item.mention_id), item.reason, rank=rank + 1,
                            required=item.required, high_changes=item.high_changes, other_changes=item.other_changes,
                            coverage_changes=item.coverage_changes, ambiguity=item.ambiguity)
            edge(mentions[item.mention_id], 'has_priority', priority)
            for ordinal, preview in enumerate(item.previews):
                if used >= preview_limit:
                    break
                used += 1
                first = tuple(ids_by_name[n] for n in preview['candidate'])
                second = tuple(ids_by_name[n] for n in preview['partner_candidate'])
                pairs = sorted(set(itertools.product(first, second)))
                possibility = node('candidate_preview', (priority, ordinal), 'Possible source findings',
                                   ordinal=ordinal, partner_mention=preview['partner_mention'],
                                   candidate_names=preview['candidate'], partner_names=preview['partner_candidate'],
                                   states=preview['states'], pair_checks_truncated=len(pairs) > MAX_PREVIEW_PAIRS,
                                   assertion='hypothetical candidate-dependent source records; not detected findings')
                edge(priority, 'has_preview', possibility)
                edge(possibility, 'assumes_candidate', candidates[(item.mention_id, tuple(preview['candidate']))])
                partner = mentions[preview['partner_mention']] if medications_by_id[preview['partner_mention']].normalized else candidates.get(
                    (preview['partner_mention'], tuple(preview['partner_candidate'])), mentions[preview['partner_mention']])
                edge(possibility, 'assumes_partner', partner)
                for a, b in pairs[:MAX_PREVIEW_PAIRS]:
                    alert = pair_alert(a, b)
                    if alert:
                        edge(possibility, 'could_retrieve', source_record(alert), assertion='hypothetical')
        graph.graph['candidate_previews_included'] = used
        graph.graph['candidate_previews_truncated'] = used < graph.graph['candidate_previews_total']
    if include_candidates:
        for assessment in result.candidate_assessments[:preview_limit]:
            item = node('candidate_assessment', (result.run_id, assessment['mention_ids']),
                        'Candidate-set assessment', status=assessment['status'],
                        stable_states=assessment['stable_states'], possible_states=assessment['possible_states'],
                        assertion='Within retained candidates only; not confirmed identities')
            edge(dossier, 'has_candidate_assessment', item)
            for mention_id in assessment['mention_ids']:
                edge(item, 'assesses_identity', mentions[mention_id])
    for ordinal, hypothesis in enumerate(result.mechanism_hypotheses):
        item = node('mechanism_hypothesis', (result.run_id, hypothesis.get('hypothesis_id', ordinal)),
                    'Mechanism-supported possibility', hypothesis=hypothesis,
                    assertion='Biological role path; not a source-severity finding')
        edge(dossier, 'has_mechanism_hypothesis', item)
        for ident in hypothesis.get('ingredient_ids', []):
            if ident in names:
                edge(item, 'hypothesis_involves', ingredient(ident))
        evidence_by_role = {e['role_id']: e for e in hypothesis.get('evidence', [])}
        for path in hypothesis.get('path', []):
            role = node('biological_role', (path['role_id'],), path['relation'],
                        role_id=path['role_id'], relation=path['relation'], conditions=hypothesis.get('conditions', []))
            target = node('biological_target', (path['target'],), path['target'])
            edge(item, 'has_biological_role', role)
            edge(role, 'role_targets', target)
            ident = ids_by_name.get(path['source'])
            subject = ingredient(ident) if ident else node('food', ('biology', path['source']), path['source'])
            edge(subject, 'plays_role', role)
            evidence = evidence_by_role.get(path['role_id'])
            if evidence:
                source = node('source_document', ('biology', evidence['url'], evidence['source_row_sha256']),
                              evidence['source'], source=evidence['source'], document_id=evidence['source_row_sha256'],
                              document_version=result.hybrid_status.get('biology', {}).get('version', ''),
                              url=evidence['url'], identifier_available=True, excerpt=evidence['excerpt'])
                edge(role, 'role_source', source)
    for ordinal, prediction in enumerate(result.model_predictions):
        item = node('model_prediction', (result.run_id, ordinal, prediction.get('ingredient_ids', [])),
                    'Experimental source-severity prediction', prediction=prediction,
                    assertion='Model output; not interaction-existence or patient-risk evidence')
        edge(dossier, 'has_model_prediction', item)
        for ident in prediction.get('ingredient_ids', []):
            if ident in names:
                edge(item, 'prediction_involves', ingredient(ident))
        if prediction.get('explanation_reference'):
            rationale = node('model_rationale', (item, prediction['explanation_reference']),
                             'Model rationale', reference=prediction['explanation_reference'])
            edge(item, 'has_model_rationale', rationale)
    graph.graph['candidate_assessments_total'] = len(result.candidate_assessments)
    graph.graph['candidate_assessments_truncated'] = include_candidates and len(result.candidate_assessments) > preview_limit
    result_graph = EvidenceGraph(graph)
    result_graph.validate()
    return result_graph


def compare_graphs(before: EvidenceGraph, after: EvidenceGraph) -> dict:
    """Compare actual source findings and their support, excluding hypothetical nodes."""
    def records(value):
        return {value.graph.nodes[n]['source_record_id']: {
            key: value.graph.nodes[n][key] for key in ('severity', 'data_version', 'ingredient_ids')}
            | {'evidence': sorted((e['evidence_id'], e['excerpt_hash'], tuple(sorted(
                (value.graph.nodes[d]['source'], value.graph.nodes[d]['document_id'],
                 value.graph.nodes[d]['document_version'], value.graph.nodes[d]['url'])
                for d in value._targets(e['node_id'], 'from_document'))))
                                  for e in value.evidence_for(value.graph.nodes[n]['source_record_id']))}
                for n in value.nodes_of_kind('finding')}
    first, second = records(before), records(after)
    return {'introduced': sorted(second.keys() - first.keys()), 'removed': sorted(first.keys() - second.keys()),
            'changed': [{'record_id': key, 'before': first[key], 'after': second[key]}
                        for key in sorted(first.keys() & second.keys()) if first[key] != second[key]],
            'scope': 'Actual source finding/evidence changes, not inferred clinical outcome changes.'}
