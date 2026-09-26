# eval/test_queries.py
# A small labelled set of test queries: each one paired with the source PDF + page that actually answers it.
# eval/run_comparison.py runs every one of these queries through all four pipeline.py variants 
# and scores the results against this ground truth.
# Every (source, page) pair below was verified directly against the real PDFs using pdfplumber — the same
# ingest.py uses to extract page text - so these page numbers will match the real chunk payloads.

TEST_QUERIES = [
    {
        "query": "How many 'building blocks' did the CPMI identify to enhance cross-border payments?",
        "relevant_sources": [
            ("enhancing-cross-border-payments-building-blocks-global-roadmap.pdf", 3),
        ],
    },
    {
        "query": "How many focus areas are the cross-border payments building blocks arranged into?",
        "relevant_sources": [
            ("enhancing-cross-border-payments-building-blocks-global-roadmap.pdf", 5),
        ],
    },
    {
        "query": "Which payment methods does Stripe recommend for SaaS and subscription businesses?",
        "relevant_sources": [
            ("Payment-methods-guide.pdf", 6),
        ],
    },
    {
        "query": "How quickly is a payment confirmed when a customer pays with Alipay?",
        "relevant_sources": [
            ("Payment-methods-guide.pdf", 12),
        ],
    },
    {
        "query": "What do the Electronic Money Regulations (EMRs) govern?",
        "relevant_sources": [
            ("payment-services-electronic-money-approach.pdf", 6),
        ],
    },
    {
        "query": "What is the temporary permissions regime (TPR) for EEA payment firms?",
        "relevant_sources": [
            ("payment-services-electronic-money-approach.pdf", 7),
        ],
    },
    {
        "query": "What are the 12 requirements of the PCI Data Security Standard?",
        "relevant_sources": [
            ("PCIDSS_QRGv3.pdf", 9),
        ],
    },
    {
        "query": "What does PCI DSS Requirement 3 say about protecting cardholder data?",
        "relevant_sources": [
            ("PCIDSS_QRGv3.pdf", 14),
        ],
    },
    {
        "query": "How does this report define a financial market infrastructure (FMI)?",
        "relevant_sources": [
            ("principles-financial-market-infrastructures.pdf", 13),
        ],
    },
    {
        "query": "Besides systemic risk, what other types of risk do financial market infrastructures face?",
        "relevant_sources": [
            ("principles-financial-market-infrastructures.pdf", 24),
        ],
    },
]