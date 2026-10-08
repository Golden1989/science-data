// Scraper de imóveis à venda em Belo Horizonte (VivaReal).
// Uso:
//   npm install playwright && npx playwright install chromium
//   node scraper.js [paginas] [arquivo.csv]
// Ex.: node scraper.js 10   (30 imóveis por página; grava em dados/brutos/imoveis.csv)
// Sem o 2º argumento, a saída vai para dados/brutos/imoveis.csv relativo a este arquivo (roda de qualquer pasta).

const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');

// Atenção: o robots.txt do site bloqueia para robôs URLs com os parâmetros onde= e tipos= (ver o README).
const BASE_URL =
  'https://www.vivareal.com.br/venda/minas-gerais/belo-horizonte/apartamento_residencial/' +
  '?onde=%2CMinas+Gerais%2CBelo+Horizonte%2C%2C%2C%2C%2Ccity%2CBR%3EMinas+Gerais%3ENULL%3EBelo+Horizonte%2C-19.919052%2C-43.938669%2C' +
  '&tipos=apartamento_residencial%2Ccasa_residencial%2Ccobertura_residencial';

const PAGINAS = parseInt(process.argv[2] || '10', 10);
const ARQUIVO = process.argv[3] || path.join(__dirname, 'dados', 'brutos', 'imoveis.csv');
// O Cloudflare do site bloqueia (403) o modo headless, por isso o padrão é abrir a janela.
// HEADLESS=1 força o modo sem janela.
const HEADLESS = process.env.HEADLESS === '1';

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const entre = (min, max) => min + Math.random() * (max - min);

// "R$ 1.530.000" -> 1530000 ; "120 - 246 m²" -> 120
const primeiroNumero = (s) => {
  const m = String(s || '').match(/\d[\d.]*(,\d+)?/);
  return m ? Number(m[0].replace(/\./g, '').replace(',', '.')) : '';
};

// Roda no navegador: extrai os dados brutos de cada cartão
function extrairCartoes(cards, pagina) {
  return cards.map((c) => {
    const t = (s) => c.querySelector(`[data-cy="${s}"]`)?.innerText.trim() ?? '';
    const ultimaLinha = (s) => t(s).split('\n').pop().trim();
    const a = c.querySelector('a');
    // Cartões "Ver os N anúncios deste imóvel" não têm link direto (href="#")
    const link = a && a.getAttribute('href') !== '#' ? a.href.split('?')[0] : '';
    const loc = t('rp-cardProperty-location-txt');
    const rua = t('rp-cardProperty-street-txt');
    // Em lançamentos, "location" traz o nome do empreendimento e "street" traz rua + bairro
    const endereco = /Belo Horizonte/.test(loc) ? (rua ? `${rua}, ${loc}` : loc) : rua;
    const m = endereco.match(/([^,]+),\s*Belo Horizonte/);
    // O título do cartão começa pelo tipo ("Apartamento para comprar com ..."); o alt da
    // imagem nem sempre (em lançamentos traz o nome do empreendimento)
    const desc = [c.querySelector('h2, h3')?.innerText, a?.title, c.querySelector('img')?.alt].join(' ');
    const tipo = (desc.match(/\b(Apartamento|Casa de Condomínio|Casa|Cobertura|Flat|Kitnet|Loft|Studio|Sobrado)\b/i) || [, ''])[1];
    const preco = t('rp-cardProperty-price-txt').split('\n').find((l) => l.includes('R$')) || '';
    return {
      pagina,
      preco,
      area: ultimaLinha('rp-cardProperty-propertyArea-txt'),
      quartos: ultimaLinha('rp-cardProperty-bedroomQuantity-txt'),
      banheiros: ultimaLinha('rp-cardProperty-bathroomQuantity-txt'),
      vagas: ultimaLinha('rp-cardProperty-parkingSpacesQuantity-txt'),
      bairro: m ? m[1].trim() : '',
      endereco,
      tipo,
      link,
    };
  });
}

// preco e area_m2 ficam como texto bruto ("R$ 1.200.000", "75 m²"); a conversão é feita na limpeza (src/limpeza.py)
function limpar(r) {
  return {
    preco: r.preco,
    area_m2: r.area,
    quartos: primeiroNumero(r.quartos),
    banheiros: primeiroNumero(r.banheiros),
    vagas: primeiroNumero(r.vagas),
    bairro: r.bairro,
    tipo: r.tipo.toLowerCase(),
    endereco: r.endereco,
    link: r.link,
    pagina: r.pagina,
  };
}

function paraCSV(linhas) {
  const cols = Object.keys(linhas[0]);
  const esc = (v) => {
    const s = String(v ?? '');
    return /[",\n;]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  return [cols.join(','), ...linhas.map((l) => cols.map((c) => esc(l[c])).join(','))].join('\n');
}

(async () => {
  const browser = await chromium.launch({ headless: HEADLESS });
  const context = await browser.newContext({
    locale: 'pt-BR',
    timezoneId: 'America/Sao_Paulo',
    viewport: { width: 1366, height: 860 },
  });
  // Versão pública: foi removido o trecho que trocava o user-agent e escondia navigator.webdriver para não
  // ser identificado como automação (contorno da proteção do site). Ver o README.
  const page = await context.newPage();
  // Fecha automaticamente modais do site (ex.: "Evoluímos a forma como mostramos...") que bloqueiam cliques
  await page.addLocatorHandler(page.locator('[role="dialog"][data-state="open"]'), async (dialog) => {
    await page.keyboard.press('Escape');
    if (await dialog.isVisible()) await dialog.getByRole('button').first().click().catch(() => {});
  });
  const brutos = [];

  for (let p = 1; p <= PAGINAS; p++) {
    console.log(`Página ${p}/${PAGINAS}...`);
    try {
      if (p === 1) {
        await page.goto(BASE_URL, { waitUntil: 'domcontentloaded', timeout: 60000 });
      } else {
        // Clica em "próxima página"; se o botão não existir, acabaram os resultados
        const proxima = page.locator('a[aria-label="próxima página"]');
        if (!(await proxima.count())) {
          console.log('  Não há próxima página.');
          break;
        }
        const anterior = await page.locator('li[data-cy="rp-property-cd"]').first().innerText();
        await proxima.first().scrollIntoViewIfNeeded();
        await sleep(entre(400, 900));
        await proxima.first().click();
        await page.waitForURL(new RegExp(`pagina=${p}(&|$)`), { timeout: 30000 });
        // Espera os cartões da nova página substituírem os anteriores
        await page.waitForFunction(
          (txt) => {
            const c = document.querySelector('li[data-cy="rp-property-cd"]');
            return c && c.innerText !== txt;
          },
          anterior,
          { timeout: 30000 }
        );
      }
      await page.waitForSelector('li[data-cy="rp-property-cd"]', { timeout: 30000 });
    } catch (e) {
      console.warn(`  Falha ao carregar a página ${p} (bloqueio ou fim dos resultados): ${e.message}`);
      break;
    }
    // Rolagem gradual, como um usuário, para carregar todos os cartões
    for (let i = 0; i < 8; i++) {
      await page.mouse.wheel(0, entre(900, 1300));
      await sleep(entre(300, 700));
    }
    const linhas = await page.$$eval('li[data-cy="rp-property-cd"]', extrairCartoes, p);
    console.log(`  ${linhas.length} imóveis`);
    brutos.push(...linhas);
    await sleep(entre(2500, 5500)); // pausa entre páginas
  }
  await browser.close();

  // Remove duplicados (mesmo link, ou mesma combinação de atributos quando não há link)
  const vistos = new Set();
  const dados = brutos.map(limpar).filter((r) => {
    const k = r.link || [r.preco, r.area_m2, r.quartos, r.banheiros, r.vagas, r.endereco].join('|');
    if (vistos.has(k)) return false;
    vistos.add(k);
    return true;
  });

  if (!dados.length) {
    console.error('Nenhum imóvel coletado.');
    process.exit(1);
  }
  fs.mkdirSync(path.dirname(ARQUIVO), { recursive: true });
  fs.writeFileSync(ARQUIVO, '\uFEFF' + paraCSV(dados), 'utf8'); // BOM para o Excel abrir acentos corretamente
  console.log(`${dados.length} imóveis salvos em ${ARQUIVO}`);
})();
