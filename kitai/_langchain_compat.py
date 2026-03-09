"""
Compatibility shim for classic LangChain imports.

In this repo's dev venv the classic LangChain framework is installed under
the alias ``langchain_classic`` to avoid collision with the Anthropic Agent
SDK (which occupies the ``langchain`` namespace at version 1.0.6).

In a standard distribution – where only classic LangChain is present as
``langchain`` – the fallback branch is taken automatically.

Invariant: all three names must be importable after this module loads,
or an ImportError propagates to the caller with a descriptive message.
"""

try:
    from langchain_classic.chains.query_constructor.schema import AttributeInfo
    from langchain_classic.retrievers.ensemble import EnsembleRetriever
    from langchain_classic.retrievers.self_query.base import SelfQueryRetriever
except ImportError:
    try:
        from langchain.chains.query_constructor.schema import AttributeInfo  # type: ignore[no-redef]
        from langchain.retrievers.ensemble import EnsembleRetriever  # type: ignore[no-redef]
        from langchain.retrievers.self_query.base import SelfQueryRetriever  # type: ignore[no-redef]
    except ImportError as exc:
        raise ImportError(
            "kitai requires classic LangChain. "
            "Install it with: pip install 'langchain>=0.1,<1.0'\n"
            "If you also use the Anthropic Agent SDK (which occupies the "
            "'langchain' namespace at v1.0.6), install classic LangChain "
            "under the alias 'langchain_classic' in your venv instead."
        ) from exc

__all__ = ["AttributeInfo", "EnsembleRetriever", "SelfQueryRetriever"]
