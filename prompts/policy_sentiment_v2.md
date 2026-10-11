# Guardian policy sentiment annotation prompt v2

You are annotating one Guardian article for a retrospective research dataset. Return one valid JSON object and no prose outside JSON. Use the exact field names and flat structure shown below.

Assess all six supplied policy topics independently. LDA probabilities are context only and never prove that a topic is present. Return exactly six topic objects, IDs 0 through 5 once each and in order. Copy each supplied Chinese `topic_name` exactly.

## Topic sentiment

- `positive`, score `1`: explicit approval, satisfaction, benefit, success, or optimism about the identified policy or condition.
- `negative`, score `-1`: explicit criticism, concern, harm, failure, burden, or pessimism about the identified policy or condition.
- `neutral`, score `0`: genuinely present but only described, without an evaluative direction.
- `mixed`, score `0`: material positive and negative evaluations coexist and neither direction clearly dominates.
- `unclear`, score `null`: evaluation exists but its object or direction cannot be determined.

Facts are not automatically sentiment. A price increase alone is neutral; an explicit statement that it makes life unaffordable is negative toward household affordability. If several objects have opposing evaluations and none dominates, use `mixed`.

## Party stance

Assess Democrats and Republicans separately and only within the current topic:

- `support`, score `1`; `oppose`, score `-1`; `neutral`, score `0`; `mixed`, score `0`;
- `not_expressed`, score `null`; `unclear`, score `null`.

Never infer support for one party from criticism of the other. Never map dissatisfaction with a condition to opposition to the governing party. Never infer party affiliation or position from outside knowledge.

## Evidence and speakers

Every non-null score requires its own short, exact, contiguous quotation from the supplied title or body, normally no more than 30 words. Do not paraphrase and do not insert ellipses. Null scores normally have an empty evidence string. A quoted criticism is a view presented by the text, not necessarily the Guardian author's view. Consider rebuttals and counter-quotes.

Use one of exactly `author`, `quoted_person`, `multiple_voices`, or `unclear` for `speaker_type`. Use `unclear` when the topic is irrelevant or no scored evidence has a speaker. Put an identifiable speaker or brief description in `speaker_name`, otherwise use an empty string.

For `relevant=false`, use exactly these defaults:

- `subissue` and `sentiment_target`: empty strings;
- `topic_sentiment_label` and `topic_sentiment_score`: null;
- both party labels: `not_expressed`; both party scores: null;
- all evidence strings empty; `speaker_type`: `unclear`; `speaker_name` empty.

Topic 2 and Topic 4 require a specific `subissue` whenever relevant. Topic 3 must distinguish war, oil price, energy supply, diplomacy, or household/business cost targets. Do not combine opposing objects without noting the ambiguity.

Return this exact shape, repeating the topic object six times with actual values:

```json
{
  "article_id": "copy the supplied article ID",
  "topics": [
    {
      "topic_id": 0,
      "topic_name": "copy supplied topic name exactly",
      "relevant": false,
      "subissue": "",
      "sentiment_target": "",
      "topic_sentiment_label": null,
      "topic_sentiment_score": null,
      "topic_sentiment_evidence": "",
      "democrat_stance_label": "not_expressed",
      "democrat_stance_score": null,
      "democrat_evidence": "",
      "republican_stance_label": "not_expressed",
      "republican_stance_score": null,
      "republican_evidence": "",
      "speaker_type": "unclear",
      "speaker_name": "",
      "ambiguity_note": ""
    }
  ]
}
```

Do not output nested evidence objects, alternative field names, comments, Markdown, election forecasts, stock returns, future information, or outside political assumptions.
