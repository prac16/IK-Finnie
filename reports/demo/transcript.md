# Finnie five-minute demo

## System design & architecture

Welcome to Finnie, a financial assistant with four focused agents. Streamlit provides the interface, while LangGraph coordinates routing, validation, specialist execution, and report composition. Each analysis starts with validated inputs and deterministic Python calculations. Only then does it retrieve financial knowledge and ask a language model to explain the results. This keeps arithmetic separate from generated prose. The knowledge base is separate from user data. Chroma retrieves document chunks, optionally filters by category, ranks twenty candidates by vector distance, and selects the top five. These are similarity results, not proof of relevance. Chat has a separate context and only uses library knowledge. Session memory holds temporary reports; explicitly saved inputs belong to a local user profile. Identifying report fields are removed before model calls, while necessary financial values remain. All supported monetary inputs use U.S. dollars.

## Chat agent · financial education

The chat agent specializes in financial education. A user can ask how diversification reduces concentration risk. Finnie retrieves library passages and generates a grounded explanation with document and page citations. Portfolio holdings and goal inputs never become chat context. When evidence is insufficient, the agent abstains instead of inventing an answer.

## Portfolio agent · concentration & exposure

The portfolio agent reads supplied holdings from CSV or Excel. It calculates total value, asset allocation, sector exposure, and concentration. Here, Tech Heavy totals three hundred twenty thousand dollars, and the largest position is thirty percent. Effective holdings are about four point four, calculated as one divided by the sum of squared portfolio weights. Findings describe the holdings and supplied goal, without fetching replacement prices.

## Market agent · observed price risk

The market agent fetches historical end-of-day data through Alpha Vantage. It calculates period return, daily changes, annualized volatility, closing-price drawdown, and benchmark differences. Volatility is the sample standard deviation of daily returns multiplied by the square root of two hundred fifty-two. Drawdown measures the largest fall from an earlier closing peak. These describe the selected historical window; they do not predict future losses or explain market news.

## Goals agent · deterministic projections

The goals agent projects savings toward a target and deadline. It converts annual growth and fee assumptions to monthly rates, applies growth and fees, then adds the monthly contribution. Today's-dollar targets are adjusted for assumed inflation. A cent-level binary search calculates the contribution needed to meet the target, with bounded deadline scenarios where allowed. In the house example, fifteen thousand dollars plus nine hundred monthly for thirty-six months reaches forty-seven thousand four hundred, leaving a twelve thousand six hundred dollar shortfall. Users can create multiple goals and analyze each separately.

## Saved profile inputs survive restart

Here is the persistence check. I saved the Tech Heavy holdings and the house-deposit goal to the Finnie Demo profile, then stopped and relaunched the Streamlit server. Selecting that profile still exposes both saved inputs, and they can be restored for a new analysis. Temporary conversations and reports are separate from these explicitly saved local inputs.

## Failure scenarios & graceful recovery

Finnie handles several failure paths. Invalid holdings or incomplete goals are rejected before calculations; the interface gives friendly guidance without exposing exception codes. Transient market API failures receive at most two retries with exponential backoff and jitter. If fetching still fails, eligible real cached data can be used with source and freshness information. Quote-level dummy defaults exist as a last resort, but the market risk engine does not use dummy history to calculate returns or volatility. If language generation or output validation fails, Finnie attempts one different model. When both attempts fail, analytical agents retain computed results and a local summary. Chat can abstain when a grounded answer cannot be prepared. Refusals are respected. Sanitized local logs record operational events without raw prompts or credentials. These recovery paths are covered by automated tests; the current suite has two hundred twenty-one passing tests.

## Future improvements

The next improvements should strengthen evidence and financial depth. First, create a labeled retrieval evaluation set and measure precision at k and recall at k, alongside answer citation accuracy and latency. Second, extend portfolio analysis with expense ratios, fund look-through, and historical correlations when reliable inputs exist. Third, add probabilistic goal simulations, retirement withdrawals, pensions, and explicit shared-budget planning across goals. Those would extend today's deterministic accumulation model. Fourth, improve market coverage with adjusted total-return data and additional providers, while keeping source dates and freshness clear. A separate news synthesizer can later explain sourced market events. Finally, production use needs authentication, encrypted storage, stronger privacy review, and operational monitoring. Finnie currently provides an educational prototype: transparent calculations, retrieved financial context, and bounded model explanations. It does not execute trades or guarantee that an investment or savings target will succeed.