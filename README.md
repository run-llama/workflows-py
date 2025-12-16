# LlamaIndex İşakışları
[![Unit Testing](https://github.com/run-llama/workflows/actions/workflows/test.yml/badge.svg)](https://github.com/run-llama/workflows/actions/workflows/test.yml)
[![Coverage Status](https://coveralls.io/repos/github/run-llama/workflows/badge.svg?branch=main)](https://coveralls.io/github/run-llama/workflows?branch=main)
[![GitHub contributors](https://img.shields.io/github/contributors/run-llama/workflows)](https://github.com/run-llama/llama-index-workflows/graphs/contributors)


[![PyPI - Downloads](https://img.shields.io/pypi/dm/llama-index-workflows)](https://pypi.org/project/llama-index-workflows/)
[![Discord](https://img.shields.io/discord/1059199217496772688)](https://discord.gg/dGcwcsnxhU)
[![Twitter](https://img.shields.io/twitter/follow/llama_index)](https://x.com/llama_index)
[![Reddit](https://img.shields.io/reddit/subreddit-subscribers/LlamaIndex?style=plastic&logo=reddit&label=r%2FLlamaIndex&labelColor=white)](https://www.reddit.com/r/LlamaIndex/)

LlamaIndex Workflows, karmaşık adım ve olay sistemlerini düzenlemek ve birbirine zincirlemek için kullanılan bir çerçevedir.

## Workflows ile neler inşa edebilirsiniz?

Workflows; yapay zeka modellerini, API'leri ve karar verme mekanizmalarını içeren karmaşık, çok adımlı süreçleri yönetmeniz gerektiğinde öne çıkar. İşte inşa edebileceklerinize dair bazı örnekler:

- **Yapay Zeka Ajanları** - Birden fazla adımda akıl yürütebilen, kararlar alabilen ve eyleme geçebilen akıllı sistemler oluşturun
- **Doküman İşleme Boru Hatları** - Dokümanları çeşitli işleme aşamalarından geçirerek alan, analiz eden, özetleyen ve yönlendiren sistemler inşa edin
- **Çok Modelli Yapay Zeka Uygulamaları** - Karmaşık görevleri çözmek için farklı yapay zeka modelleri (LLM'ler, görü modelleri vb.) arasında koordinasyon sağlayın
- **Araştırma Asistanları** - Bilgiyi arayabilen, analiz edebilen, sentezleyebilen ve kapsamlı yanıtlar sağlayabilen iş akışları geliştirin
- **İçerik Üretim Sistemleri** - İnsan onaylı süreçlerle içerik üreten, gözden geçiren, düzenleyen ve yayınlayan boru hatları oluşturun
- **Müşteri Destek Otomasyonu** - Müşteri taleplerini anlayabilen, kategorize edebilen ve yanıtlayabilen akıllı yönlendirme sistemleri inşa edin

Asenkron öncelikli (async-first), olay güdümlü (event-driven) mimari; farklı yetenekler arasında yönlendirme yapabilen, paralel işleme modelleri uygulayabilen, karmaşık diziler üzerinde döngü kurabilen ve birden fazla adımda durumu (state) koruyabilen iş akışları oluşturmayı kolaylaştırır - yapay zeka uygulamalarınızı üretime hazır hale getirmek için ihtiyacınız olan tüm özellikler.

## Temel Özellikler

- **async-first (asenkron öncelikli)** - iş akışları Python'un async işlevselliği etrafında oluşturulmuştur; adımlar, bir asyncio kuyruğundan gelen olayları işleyen ve diğer kuyruklara yeni olaylar yayan async fonksiyonlardır. Bu aynı zamanda iş akışlarının FastAPI, Jupyter Notebook vb. gibi async uygulamalarınızda en iyi şekilde çalıştığı anlamına gelir.
- **event-driven (olay güdümlü)** - iş akışları adımlardan ve olaylardan oluşur. Kodunuzu olaylar ve adımlar etrafında organize etmek, mantık yürütmeyi ve test etmeyi kolaylaştırır.
- **state management (durum yönetimi)** - bir iş akışının her çalıştırılması kendi içinde bağımsızdır; yani bir iş akışını başlatabilir, içine bilgi kaydedebilir, iş akışının durumunu serileştirebilir ve daha sonra devam ettirebilirsiniz.
- **observability (gözlemlenebilirlik)** - iş akışları gözlemlenebilirlik için otomatik olarak donatılmıştır, yani `Arize Phoenix` ve `OpenTelemetry` gibi araçları kurulum gerektirmeden doğrudan kullanabilirsiniz.

## Hızlı Başlangıç

Paketi yükleyin:

```bash
pip install llama-index-workflows
```

Ve ilk iş akışınızı oluşturun:

```python
import asyncio
from pydantic import BaseModel, Field
from workflows import Context, Workflow, step
from workflows.events import Event, StartEvent, StopEvent

class MyEvent(Event):
    msg: list[str]

class RunState(BaseModel):
    num_runs: int = Field(default=0)

class MyWorkflow(Workflow):
    @step
    async def start(self, ctx: Context[RunState], ev: StartEvent) -> MyEvent:
        async with ctx.store.edit_state() as state:
            state.num_runs += 1

            return MyEvent(msg=[ev.input_msg] * state.num_runs)

    @step
    async def process(self, ctx: Context[RunState], ev: MyEvent) -> StopEvent:
        data_length = len("".join(ev.msg))
        new_msg = f"Processed {len(ev.msg)} times, data length: {data_length}"
        return StopEvent(result=new_msg)

async def main():
    workflow = MyWorkflow()

    # [optional] provide a context object to the workflow
    ctx = Context(workflow)
    result = await workflow.run(input_msg="Hello, world!", ctx=ctx)
    print("Workflow result:", result)

    # re-running with the same context will retain the state
    result = await workflow.run(input_msg="Hello, world!", ctx=ctx)
    print("Workflow result:", result)


if __name__ == "__main__":
    asyncio.run(main())
```

Yukarıdaki örnekte
- `StartEvent` kabul eden adımlar ilk olarak çalıştırılır.
- `StopEvent` döndüren adımlar iş akışını sonlandırır.
- Ara olaylar kullanıcı tanımlıdır ve adımlar arasında bilgi aktarmak için kullanılabilir.
- `Context` nesnesi de adımlar arasında bilgi paylaşmak için kullanılır.

`llama-index` kullanan daha fazla örnek için [kapsamlı dokümantasyonu](https://docs.llamaindex.ai/en/stable/understanding/workflows/) ziyaret edin!

## Daha Fazla Örnek

- [Basic Feature Run-Through](./examples/feature_walkthrough.ipynb)
- [Building a Function Calling Agent with `llama-index`](./examples/agent.ipynb)
- [Human-in-the-loop Iterative Document Extraction](./examples/document_processing.ipynb)
- Observability
  - [OpenTelemetry + Instrumentation Primer](./examples/observability/workflows_observability_pt1.ipynb)
  - [OpenTelemetry + LlamaIndex](./examples/observability/workflows_observability_pt2.ipynb)
  - [Arize Phoenix + LlamaIndex](./examples/observability/workflows_observablitiy_arize_phoenix.ipynb)
  - [Langfuse + LlamaIndex](./examples/observability/workflows_observablitiy_langfuse.ipynb)

## İlgili Paketler

- [Typescript Workflows](https://github.com/run-llama/workflows-ts)
