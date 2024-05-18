// @ts-check
// `@type` JSDoc annotations allow editor autocompletion and type checking
// (when paired with `@ts-check`).
// There are various equivalent ways to declare your Docusaurus config.
// See: https://docusaurus.io/docs/api/docusaurus-config

import {themes as prismThemes} from 'prism-react-renderer';
import remarkDefList from 'remark-deflist';
import tabBlocks from "docusaurus-remark-plugin-tab-blocks";

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'Infernal',
  tagline: 'Production-quality edge inference',
  favicon: 'img/favicon.ico',

  // Set the production url of your site here
  url: 'https://infernal.rs',
  // Set the /<baseUrl>/ pathname under which your site is served
  // For GitHub pages deployment, it is often '/<projectName>/'
  baseUrl: '/',

  // GitHub pages deployment config.
  // If you aren't using GitHub pages, you don't need these.
  organizationName: 'HaltiaAI', // Usually your GitHub org/user name.
  projectName: 'infernal.rs', // Usually your repo name.

  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',

  // Even if you don't use internationalization, you can use this field to set
  // useful metadata like html lang. For example, if your site is Chinese, you
  // may want to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          routeBasePath: '/', // Serve the docs at the site's root
          sidebarPath: './sidebars.js',
          editUrl:
            'https://github.com/HaltiaAI/infernal.rs/tree/master/packages/create-docusaurus/templates/shared/',
          remarkPlugins: [remarkDefList, [tabBlocks, {
            labels: [
              ["bash", "CLI"],
              ["shell-session", "Shell"],
            ],
          }]],
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
      }),
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      image: 'img/docusaurus-social-card.jpg', // TODO
      navbar: {
        title: 'Infernal',
        logo: {
          alt: 'Infernal Logo', // TODO
          src: 'img/logo.svg',
        },
        items: [
          {
            type: 'docSidebar',
            sidebarId: 'home',
            label: 'Overview',
            position: 'left',
          },
          {
            type: 'docSidebar',
            sidebarId: 'install',
            label: 'Install',
            position: 'left',
          },
          {
            type: 'docSidebar',
            sidebarId: 'learn',
            label: 'Learn',
            position: 'left',
          },
          {
            type: 'docSidebar',
            sidebarId: 'api',
            label: 'API',
            position: 'left',
          },
          {
            href: 'https://github.com/HaltiaAI/infernal.rs',
            label: 'GitHub',
            position: 'right',
          },
        ],
      },
      footer: {
        style: 'dark',
        links: [
          {
            title: 'Docs',
            items: [
              {
                label: 'Overview',
                to: '/',
              },
              {
                label: 'Install',
                to: '/install',
              },
              {
                label: 'Learn',
                to: '/concepts',
              },
              {
                label: 'API',
                to: '/api/',
              },
            ],
          },
          {
            title: 'Community',
            items: [
              {
                label: 'Stack Overflow',
                href: 'https://stackoverflow.com/questions/tagged/infernal',
              },
              {
                label: 'Twitter',
                href: 'https://twitter.com/HaltiaEngineers',
              },
            ],
          },
          {
            title: 'More',
            items: [
              {
                label: 'GitHub',
                href: 'https://github.com/HaltiaAI/infernal.rs',
              },
            ],
          },
        ],
        copyright: `Copyright © 2023 Haltia, Inc.`,
      },
      prism: {
        theme: prismThemes.github,
        darkTheme: prismThemes.dracula,
        defaultLanguage: 'shell-session',
        additionalLanguages: ['bash', 'json', 'python', 'rust', 'shell-session'],
      },
    }),
  plugins: [
    () => ({
      name: 'symlink-resolver',
      configureWebpack() {
        return { resolve: { symlinks: false } };
      },
    }),
    ['@grnet/docusaurus-terminology', {
      termsDir: './docs/glossary',
      docsDir: './docs/',
      glossaryFilepath: './docs/glossary.md'
    }],
  ]
};

export default config;
