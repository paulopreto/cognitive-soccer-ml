# [As funções cognitivas são capazes de predizer a efetividade geral e as contribuições do atleta para a equipe em jogos reduzidos de futebol? Uma abordagem utilizando aprendizado de máquina.](https://bv.fapesp.br/pt/bolsas/209042/as-funcoes-cognitivas-sao-capazes-de-predizer-a-efetividade-geral-e-contribuicao-do-atleta-para-a-eq/)

## Visão geral do projeto

Este repositório contém dados e códigos desenvolvidos em Python para a predição do desempenho de jovens jogadores em jogos reduzidos por meio da análise de funções cognitivas utilizando algoritmos de aprendizado de máquina. Os dados compartilhados são resultado do projeto de mestrado do aluno Rafael Luiz Martins Monteiro com fomento da Fundação de Amparo à Pesquisa do Estado de São Paulo, processo #2021/15134-9. A dissertação é intitulada: "As funções cognitivas são capazes de predizer a efetividade geral e as contribuições do atleta para a equipe em jogos reduzidos de futebol? Uma abordagem utilizando aprendizado de máquina.".

Foram utilizados dados obtidos por meio de testes computadorizados com softwares como [PEBL](https://pebl.sourceforge.net/) e [PsychoPy](https://www.psychopy.org/), focando em avaliações de atenção sustentada, memória visuoespacial, impulsividade, flexibilidade cognitiva e capacidade de rastreamento de objetos múltiplos. Os testes utilizados foram: Blocos de Corsi, Go/No Go e teste de trilhas no PEBL e rastreamento de objetos múltiplos no PsychoPy.

A avaliação em campo foi feita por meio de jogos reduzidos de 3 x 3 jogadores sem goleiros. Cada jogo tinha duração de 4 minutos. Após cada rodada os times eram misturados de forma aleatória, dessa forma os jogadores não jogavam com ou contra os mesmos colegas. O objetivo da avaliação foi extrair a performance individual de cada atleta por meio da interação coletiva entre os jogadores. Este protocolo foi proposto e validado por [Wilson et al. 2021](https://onlinelibrary.wiley.com/doi/full/10.1111/sms.13969).

## Configuração para desenvolvimento (Linux, Windows, macOS)

O projeto usa [uv](https://docs.astral.sh/uv/) com Python 3.12.12. Na raiz do projeto: **Linux/macOS** — `bash install.sh` (instala uv, cria `.venv`, instala dependências e testa). **Windows** — instale o uv e rode `uv venv --python 3.12.12 --clear` e depois `uv sync`. Para executar o pipeline completo: `uv run python run_pipeline.py`. Para apenas treino: `uv run python -m src.train_final_models`. Para apenas o teste de permutação (validação de overfitting): `uv run python -m src.validate_overfitting`.

## Acesso aos dados

Todos os códigos utilizados para o processamento de dados podem ser acessados na pasta [src](../src/). Cada código possui uma breve descrição de suas funcionalidades e de como utilizá-lo. Os dados anonimizados utilizados para o processamento e trabalho final podem ser acessados na pasta [data](../data/). Dentro desta pasta tem os dados brutos e os dados processados que foram utilizados para análise no arquivo `dataset.csv`. Os melhores hiperparâmetros resultantes do GridSearch podem ser encontrados na pasta [best_param](../best_param/). Os resultados do treinamento dos modelos de aprendizado de máquina assim como as análises estatísticas podem ser acessados na pasta [results_CV](../results_CV/). As figuras do artigo podem ser acessadas na pasta [figures](../figures/).

### Agradecimentos

Os autores agradecem à comissão técnica e aos atletas pela participação e contribuições fundamentais para este estudo. Além disso, agradecemos o apoio financeiro da Fundação de Amparo à Pesquisa do Estado de São Paulo (FAPESP), processos: [#2024/17521-8](https://bv.fapesp.br/pt/bolsas/224064/analise-do-desenvolvimento-cognitivo-na-predicao-de-variaveis-tecnicas-taticas-e-fisicas-de-jogadore/), [#2024/15658-6](https://bv.fapesp.br/pt/bolsas/224975/classificacao-de-esforcos-em-corrida-com-aprendizado-maquina-por-meio-da-forca-de-reacao-do-solo-e-r/), [#2021/15134-9](https://bv.fapesp.br/pt/bolsas/209042/as-funcoes-cognitivas-sao-capazes-de-predizer-a-efetividade-geral-e-contribuicao-do-atleta-para-a-eq/), [#2019/22262-3](https://bv.fapesp.br/51740), e [#2019/17729-0](https://bv.fapesp.br/51021). Também agradecemos à Coordenação de Aperfeiçoamento de Pessoal de Nível Superior (CAPES) – Código de Financiamento 001, e ao Centro Acadêmico Bavariano para a América Latina (BAYLAT).

## Contato

Para mais informações ou dúvidas, por favor, entre em contato pelo e-mail [rafaell_mmonteiro@usp.br](mailto:rafaell_mmonteiro@usp.br).
