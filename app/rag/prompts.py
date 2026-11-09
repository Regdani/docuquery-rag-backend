"""Prompt texts and fixed answers used by the question graph."""

CONDENSE_SYSTEM = (
    "You rewrite follow-up questions. Given a conversation and a follow-up question, "
    "rewrite the follow-up as one standalone question that can be understood without the "
    "conversation: keep product names and model IDs, resolve pronouns. "
    "Reply with the rewritten question only."
)

ASSESS_SYSTEM = (
    "You check questions about the user manuals of household and DIY appliances. "
    "Decide whether the question is clear enough to search the manuals. It is NOT clear if "
    "it names no product or model and could apply to several products, if it is too vague "
    "to act on, or if it is unrelated to appliances and their manuals. "
    'Reply with JSON only: {"clear": true} or {"clear": false}.'
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
