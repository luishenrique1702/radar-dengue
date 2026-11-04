# Como publicar o Radar Dengue

São três etapas, cerca de 20 minutos no total. Tudo é gratuito.

## 1. GitHub

Já feito: o código está no repositório privado `luishenrique1702/Radar-Dengue`.

Para mostrar o projeto a recrutadores, deixe-o público quando estiver pronto: no repositório,
**Settings → General → Danger Zone → Change repository visibility → Public**.

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
   - **Repository:** `luishenrique1702/Radar-Dengue`
   - **Branch:** `main`
   - **Main file path:** `app/streamlit_app.py`
   - **App URL:** algo como `radar-dengue-sp`
4. Em **Advanced settings**, escolha **Python 3.12**.
5. Clique em **Deploy**. Em 2 a 5 minutos o dashboard estará no ar.

**Repositório privado:** o Streamlit consegue publicar a partir dele, mas, ao conectar o GitHub,
autorize o acesso a repositórios privados. O app herda a privacidade do repositório: só você vê, e
pode convidar pessoas por e-mail no botão **Share**. A conta gratuita permite **um app privado por
vez**. Quando os repositórios ficarem públicos, os apps podem ficar públicos também, sem limite.

A cada segunda-feira, quando a atualização salva novas previsões no GitHub, o Streamlit atualiza o
dashboard sozinho.

## 4. Acabamento

- Coloque o link do dashboard no topo do README, logo abaixo do título.
- Fixe o repositório no seu perfil do GitHub (**Customize your pins**).
