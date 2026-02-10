# [Are Cognitive Functions Capable of Predicting Overall Effectiveness and Player Contributions in Small-Sided Soccer Games? A Machine Learning Approach](https://bv.fapesp.br/en/scholarships/209042/are-cognitive-functions-capable-of-predicting-overall-effectiveness-and-player-contributions-in-sma/)

## Project Overview

This repository contains data and Python scripts developed for predicting the performance of young soccer players in small-sided games through the analysis of cognitive functions using machine learning algorithms. The shared data is part of the master’s thesis project of student Rafael Luiz Martins Monteiro, funded by the São Paulo Research Foundation (FAPESP), grant #2021/15134-9. The dissertation is entitled: **"Are cognitive functions capable of predicting overall effectiveness and player contributions in small-sided soccer games? A machine learning approach."**

Data was collected using computerized tests with software such as [PEBL](https://pebl.sourceforge.net/) and [PsychoPy](https://www.psychopy.org/), focusing on assessments of sustained attention, visuospatial memory, impulsivity, cognitive flexibility, and multiple object tracking capacity. The tests used included the **Corsi Block-Tapping Test**, **Go/No-Go**, **Trail Making Test (PEBL)**, and **Multiple Object Tracking (PsychoPy)**.

On-field evaluation was performed through **3 vs. 3 small-sided games without goalkeepers**. Each match lasted 4 minutes. After every round, teams were randomly shuffled, ensuring that players did not repeatedly play with or against the same teammates. The objective of this evaluation was to extract each athlete’s individual performance through collective interactions among players. This protocol was proposed and validated by [Wilson et al. 2021](https://onlinelibrary.wiley.com/doi/full/10.1111/sms.13969).

<p align="center">
  <img src="./figures/Figure_1.png" alt="Field" />
</p>

<p align="center">
  <img src="./figures/Figure_3.png" alt="Graph" />
</p>


## Data Access

All scripts used for data processing are available in the [src](./src/) folder. Each script contains a brief description of its functionality and usage instructions. Always remember to adjust file paths according to your local setup.

The anonymized data used for processing and final analyses are available in the [data](./data/) folder. This folder contains both **raw data** and **processed data** used in the file `dataset.csv`.

The best hyperparameters obtained from GridSearch are provided in the [best_param](./best_param/) folder.

Machine learning training results and statistical analyses are available in the [results_CV](./results_CV/) folder.

Figures used in the publication can be accessed in the [figures](./figures/) folder.

## Acknowledgements

The authors would like to thank the coaching staff and athletes for their participation and valuable contributions to this study. We also acknowledge the financial support from the São Paulo Research Foundation (FAPESP), grants:

- [#2024/17521-8](https://bv.fapesp.br/en/scholarships/224064/)
- [#2024/15658-6](https://bv.fapesp.br/en/scholarships/224975/)
- [#2021/15134-9](https://bv.fapesp.br/en/scholarships/209042/)
- [#2019/22262-3](https://bv.fapesp.br/51740)
- [#2019/17729-0](https://bv.fapesp.br/51021)

We also thank the Coordination for the Coordenação de Aperfeiçoamento de Pessoal de Nível Superior (CAPES) – Funding Code 001, and the Bavarian Academic Center for Latin America (BAYLAT).

## Contact

For further information or questions, please contact [rafaell_mmonteiro@usp.br](mailto:rafaell_mmonteiro@usp.br).


# [As funções cognitivas são capazes de predizer a efetividade geral e as contribuições do atleta para a equipe em jogos reduzidos de futebol? Uma abordagem utilizando aprendizado de máquina.](https://bv.fapesp.br/pt/bolsas/209042/as-funcoes-cognitivas-sao-capazes-de-predizer-a-efetividade-geral-e-contribuicao-do-atleta-para-a-eq/)

## Visão geral do projeto

Este repositório contém dados e códigos desenvolvidos em Python para a predição do desempenho de jovens jogadores em jogos reduzidos por meio da análise de funções cognitivas utilizando algoritmos de aprendizado de máquina. Os dados compartilhados são resultado do projeto de mestrado do aluno Rafael Luiz Martins Monteiro com fomento da Fundação de Amparo à Pesquisa do Estado de São Paulo, processo #2021/15134-9. A dissertação é intitulada: "As funções cognitivas são capazes de predizer a efetividade geral e as contribuições do atleta para a equipe em jogos reduzidos de futebol? Uma abordagem utilizando aprendizado de máquina.".

Foram utilizados dados obtidos por meio de testes computadorizados com softwares como PEBL (https://pebl.sourceforge.net/)  e PsychoPy (https://www.psychopy.org/), focando em avaliações de atenção sustentada, memória visuoespacial, impulsividade, flexibilidade cognitiva e capacidade de rastreamento de objetos múltiplos. Os testes utilizados foram: Blocos de Corsi, Go/ No Go e teste de trilhas no PEBL e rastreamento de objetos múltiplos no PsychoPy.

A avaliação em campo foi feita por meio de jogos reduzidos de 3 x 3 jogadores sem goleiros. Cada jogo tinha duração de 4 minutos. Após cada rodada os times eram misturados de forma aleatória, dessa forma os jogadores não jogavam com ou contra os mesmos colegas. O objetivo da avaliação foi extrair a performance individual de cada atleta por meio da interação coletiva entre os jogadores. Este protoclo foi proposto e validado por Wilson et al. 2021 (https://onlinelibrary.wiley.com/doi/full/10.1111/sms.13969).


## Acesso aos dados
Todos os códigos utilizados para o processamento de dados podem ser acessados na pasta [src](./src/). Cada código possui uma breve descrição de suas funcionalidades e de como utiliza-lo. Lembre-se sempre de ajustar os caminhos dos arquivos para executa-los em seu próprio computador. Os dados anonimizados utilizados para o processamento e trabalho final podem ser acessados na pasta [data](./data/). Dentro desta pasta tem os dados brutos e os dados processados que foram utilizados para análise no arquivo 'dataset.csv'. Os melhores hiperparâmetros resultantes do GridSearch podem ser encontrados na pasta [best_param](./best_param/). Os resultados do treinamento dos modelos de aprendizado de máquina assim como as análises estatísticas podem ser acessados na pasta [results_CV](./results_CV/). As figuras do artigo podem ser acessadas na pasta [figures](./figures/). 


### Agradecimentos
Os autores agradecem à comissão técnica e aos atletas pela participação e contribuições fundamentais para este estudo. Além disso, agradecemos o apoio financeiro da Fundação de Amparo à Pesquisa do Estado de São Paulo (FAPESP), processos: [#2024/17521-8](https://bv.fapesp.br/pt/bolsas/224064/analise-do-desenvolvimento-cognitivo-na-predicao-de-variaveis-tecnicas-taticas-e-fisicas-de-jogadore/), [#2024/15658-6](https://bv.fapesp.br/pt/bolsas/224975/classificacao-de-esforcos-em-corrida-com-aprendizado-maquina-por-meio-da-forca-de-reacao-do-solo-e-r/), [#2021/15134-9](https://bv.fapesp.br/pt/bolsas/209042/as-funcoes-cognitivas-sao-capazes-de-predizer-a-efetividade-geral-e-contribuicao-do-atleta-para-a-eq/), [#2019/22262-3](https://bv.fapesp.br/51740), e [#2019/17729-0](https://bv.fapesp.br/51021). Também agradecemos à Coordenação de Aperfeiçoamento de Pessoal de Nível Superior (CAPES) – Código de Financiamento 001, e ao Centro Acadêmico Bavariano para a América Latina (BAYLAT).

## Contato
Para mais informações ou dúvidas, por favor, entre em contato pelo e-mail [rafaell_mmonteiro@usp.br](mailto:rafaell_mmonteiro@usp.br).

