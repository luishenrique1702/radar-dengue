# Como publicar o Radar Dengue

São três etapas, cerca de 20 minutos no total. Tudo é gratuito.

## 1. Colocar o código no GitHub

1. Crie uma conta em [github.com](https://github.com), se ainda não tiver.
2. Clique em **New repository**, dê o nome `radar-dengue`, deixe **Public** e **não** marque a
   opção de criar README (o projeto já tem um).
3. Envie o código. Escolha uma das formas:
   - **Pelo terminal**, dentro da pasta do projeto:
     ```bash
     git remote add origin https://github.com/SEU-USUARIO/radar-dengue.git
     git push -u origin main
     ```
   - **Pelo VS Code:** abra a pasta, vá em *Source Control* e clique em *Publish Branch*.

## 2. Ligar a atualização automática

1. No repositório, vá em **Settings → Actions → General**.
2. Em **Workflow permissions**, marque **Read and write permissions** e salve. Isso deixa a
   atualização semanal salvar as previsões novas.
3. Vá em **Actions → Atualização semanal → Run workflow** para rodar a primeira vez (cerca de 10
   minutos). Depois ela roda sozinha toda segunda-feira.

Se algo falhar, ou o modelo piorar, o GitHub envia um e-mail.

## 3. Publicar o dashboard

1. Entre em [share.streamlit.io](https://share.streamlit.io) com a sua conta do GitHub.
2. Clique em **Create app → Deploy a public app from GitHub**.
3. Preencha:
   - **Repository:** `SEU-USUARIO/radar-dengue`
   - **Branch:** `main`
   - **Main file path:** `app/streamlit_app.py`
   - **App URL:** algo como `radar-dengue-sp`
4. Em **Advanced settings**, escolha **Python 3.12**.
5. Clique em **Deploy**. Em 2 a 5 minutos o dashboard estará no ar.

A cada segunda-feira, quando a atualização salva novas previsões no GitHub, o Streamlit atualiza o
dashboard sozinho.

## 4. Acabamento

- Troque *link do Streamlit Cloud aqui* no README pelo endereço do dashboard.
- Troque *SEU NOME* e os links do LinkedIn e do Lattes.
- Fixe o repositório no seu perfil do GitHub (**Customize your pins**).
