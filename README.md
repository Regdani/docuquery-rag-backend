# DocuQuery – RAG backend

## Problem

A large webshop sells household and DIY appliances, and every product comes with a user manual. These manuals run to many pages, so customers and customer support agents struggle to find specific answers in them. The target users are customers before a purchase, who compare product features and technical data; customers after a purchase, who turn to the manual with questions about usage, maintenance or troubleshooting; and customer support agents, who want to give accurate answers quickly. DocuQuery is an HTTP API that the webshop's search box can be built on. Users ask their question in natural language, for example: "How often should I descale the KV-200 coffee maker?" The system answers from the relevant passages of the manuals rather than from a manual as a whole. The answer also names its source, meaning the product and the manual section. This lets users verify the answer without having to read the entire document.

### Example questions

- **Usage:** How do I start the espresso program on the KV-200 coffee maker?
- **Maintenance:** How often should I descale the KV-200 coffee maker?
- **Troubleshooting:** Why is the red light flashing on the drill, and what should I do?
- **Technical data:** What are the maximum speed and the power of the impact drill?
- **Safety:** Can I use the chainsaw in the rain, and what protective equipment do I need?

### Scope and limitations

- Answers come exclusively from the user manuals.
- Price, stock, orders and product recommendations are not part of the application.
- The webshop's search interface (UI) is out of scope; the prototype provides only the API.
- There is no authentication.
- There is no quality measurement (answer quality is not evaluated).
- Conversation history is kept per `session_key`, in memory only, and is lost on restart.

## Sample data

Real manufacturer manuals are copyrighted, so the repository ships its own short manuals in `data/sample/`. All products and brands in them (KV-200, KV-300, DR-450, CS-350, RV-700, WM-800, AP-120) are fictional, and the documents contain no confidential or real manufacturer content. Each manual is a markdown file with the same sections: Safety, Setup, Usage, Maintenance, Troubleshooting and Technical data. The set includes two similar coffee makers (KV-200 and KV-300) with different descaling intervals, so retrieval has to pick the right product.

### Questions the documents can answer

- How do I start the espresso program on the KV-200 coffee maker? (`kv-200-espresso-coffee-maker.md`, Usage)
- How often should I descale the KV-200 coffee maker? (`kv-200-espresso-coffee-maker.md`, Maintenance: every 60 days or 200 brew cycles. The KV-300 drip coffee maker differs: every 30 days.)
- Why is the red light flashing on the DR-450 drill, and what should I do? (`dr-450-cordless-impact-drill.md`, Troubleshooting)
- What are the maximum speed and the power of the DR-450 impact drill? (`dr-450-cordless-impact-drill.md`, Technical data: 2000 rpm, 600 W)
- Can I use the CS-350 chainsaw in the rain, and what protective equipment do I need? (`cs-350-electric-chainsaw.md`, Safety)

### Questions the documents cannot answer

- How much does the KV-200 coffee maker cost?
- Is the WM-800 washing machine in stock?
- How do I descale the KV-500 coffee maker? (this product is not in the set)
- Which smartphones can control the RV-700 robot vacuum? (no manual states this)
