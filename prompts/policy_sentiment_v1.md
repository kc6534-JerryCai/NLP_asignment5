# Guardian policy sentiment annotation prompt v1

You are annotating one Guardian article for a retrospective research dataset. Return one JSON object and no prose outside JSON.

The article must be assessed independently for all six supplied policy topics. LDA probabilities are context only and never prove that a topic is present. For a topic that is not genuinely discussed, set `relevant=false`, `topic_sentiment_label=null`, all three scores to null, and both party labels to `not_expressed`.

## Topic sentiment

- `positive`, score `1`: explicit approval, satisfaction, benefit, success, or optimism about the identified policy or condition.
- `negative`, score `-1`: explicit criticism, concern, harm, failure, burden, or pessimism about the identified policy or condition.
- `neutral`, score `0`: the topic is genuinely present but the text only describes it without an evaluative direction.
- `mixed`, score `0`: material positive and negative evaluations coexist and no direction clearly dominates.
- `unclear`, score `null`: evaluation exists but its object or direction cannot be determined.

Facts are not automatically sentiment. A price increase alone is neutral; an explicit statement that it makes life unaffordable is negative toward household affordability. When a topic has multiple evaluated objects, list them briefly and score the clearly dominant evaluation. If no evaluation dominates, use `mixed`.

## Party stance

Assess Democrats and Republicans independently and only within the current topic.

- `support`, score `1`: explicit support or praise.
- `oppose`, score `-1`: explicit opposition or criticism.
- `neutral`, score `0`: the party, politician, or party policy is explicitly involved but no evaluative direction is expressed.
- `mixed`, score `0`: support and criticism coexist without a dominant direction.
- `not_expressed`, score `null`: no identifiable topic-specific stance toward that party.
- `unclear`, score `null`: a party-related stance appears but its target or direction cannot be determined.

Never infer support for one party from criticism of the other. Never map dissatisfaction with inflation or another condition to opposition to the governing party. Never infer a party position merely because a policy is commonly associated with that party. Party or candidate affiliation must be explicit in the article.

## Evidence and voices

For every non-null score, provide a short, exact, contiguous quotation from the supplied title or body, normally no more than 30 words. Do not use ellipses or paraphrase inside an evidence field. For null scores, leave evidence empty unless the label is `unclear`, where a short exact quotation may explain the ambiguity.

Distinguish author or editorial evaluation from quoted people and competing voices. Use `author`, `quoted_person`, `multiple_voices`, or `unclear` for each evidence speaker type and give a speaker name or description where identifiable. A quoted criticism is a criticism presented by the text, not necessarily the Guardian reporter's own view. Consider rebuttals and counter-quotes before choosing a direction.

## Output constraints

- Return exactly six topic objects, topic IDs 0 through 5 once each, in order.
- Copy each supplied topic name exactly.
- `subissue` and `sentiment_target` are concise strings. Topic 2 and Topic 4 require a specific subissue whenever relevant. Topic 3 must distinguish war, oil price, and energy-supply targets.
- Do not use election forecasts, stock returns, future information, or outside political assumptions.
- The JSON top level is `{"article_id": "...", "topics": [...]}`.

