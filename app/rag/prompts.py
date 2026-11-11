"""Prompt texts and fixed answers used by the question graph."""

_CLEAR_CRITERIA = (
    "Model IDs such as KV-200 or DR-450 are the webshop's own products; their manuals are "
    "available, so never ask about the brand or manufacturer. A question that names a model ID "
    "or comes with a 'Product:' line is clear, even if it is short or says 'it'. "
    "It is NOT clear if it names no product or model and could apply to several products, "
    "if it is too vague to act on, or if it is unrelated to appliances and their manuals."
)

ASSESS_SYSTEM = (
    "You check questions about the user manuals of household and DIY appliances. "
    f"Decide whether the question is clear enough to search the manuals. {_CLEAR_CRITERIA} "
    'Reply with JSON only: {"clear": true} or {"clear": false}.'
)

ANALYZE_SYSTEM = (
    "You rewrite follow-up questions about the user manuals of household and DIY appliances. "
    "Given a conversation and a follow-up question, first rewrite the follow-up as one "
    "standalone question that can be understood without the conversation: keep product names "
    "and model IDs, resolve pronouns. Then decide whether that standalone question is clear "
    f"enough to search the manuals. {_CLEAR_CRITERIA} "
    'Reply with JSON only: {"question": "<standalone question>", "clear": true} '
    '(or "clear": false).'
)

CLARIFY_SYSTEM = (
    "The user asked a question about appliance manuals that cannot be answered yet because it "
    "is incomplete, ambiguous or unrelated to the manuals. Ask ONE short, targeted clarifying "
    "question (for example which product or model they mean). Reply with that question only."
)

ANSWER_SYSTEM = (
    "You answer questions about appliance user manuals. Use ONLY the context below. "
    "Do not use outside knowledge. If the context does not contain the answer, say that you "
    "do not know based on the manuals. Be concise.\n\nContext:\n{context}"
)

NO_ANSWER_TEXT = (
    "I could not find relevant information about this in the available manuals. "
    "Please rephrase the question or name the product you mean."
)
