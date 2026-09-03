import { defineConfig } from 'vitepress'

// Project site on GitHub Pages, so every absolute path is prefixed with the repo name.
// Change `base` to '/' if this ever moves to a user site or a custom domain.
const base = '/harbor-rl/'

// og:image must be an ABSOLUTE url. Social scrapers fetch the tag without a document base
// to resolve against, so a site-root path like `/harbor-rl/social-preview.png` is simply
// dropped and the card renders blank — the one bug you cannot see from your own browser.
// Keep this in step with `base` if the site moves.
const siteUrl = 'https://supersglzc.github.io/harbor-rl/'

export default defineConfig({
  title: 'HARBOR',
  description: 'A harness framework for agentic robot reinforcement learning. Point it at a simulator, describe a task, get a trained policy.',
  base,
  lang: 'en-US',
  cleanUrls: true,
  lastUpdated: true,
  ignoreDeadLinks: false,

  // The concepts page states the MDP and the harness tuple; both read far better set than
  // spelled out in prose. Requires markdown-it-mathjax3, which package.json pins.
  markdown: { math: true },

  head: [
    ['link', { rel: 'icon', type: 'image/svg+xml', href: `${base}favicon.svg` }],
    ['meta', { property: 'og:type', content: 'website' }],
    ['meta', { property: 'og:title', content: 'HARBOR — agentic robot reinforcement learning' }],
    ['meta', { property: 'og:description', content: 'Point it at a simulator. Describe a task. Get a trained policy.' }],
    ['meta', { property: 'og:url', content: siteUrl }],
    ['meta', { property: 'og:image', content: `${siteUrl}social-preview.png` }],
    ['meta', { name: 'twitter:card', content: 'summary_large_image' }],
    ['meta', { name: 'twitter:image', content: `${siteUrl}social-preview.png` }],
  ],

  themeConfig: {
    logo: { light: '/logo.svg', dark: '/logo-dark.svg' },
    siteTitle: 'HARBOR',

    // One entry per sidebar section, each a dropdown of that section's pages. The sidebar
    // is still the full map; the nav exists so a reader landing mid-site can jump straight
    // to a section without scrolling the sidebar to find where they are.
    nav: [
      {
        text: 'Getting started',
        activeMatch: '^/guide/(install|first-benchmark)?$',
        items: [
          { text: 'What is HARBOR?', link: '/guide/' },
          { text: 'Installation', link: '/guide/install' },
          { text: 'Your first benchmark', link: '/guide/first-benchmark' },
        ],
      },
      {
        text: 'Workflows',
        activeMatch: '^/guide/(end-to-end|tasks|rewards|training)$',
        items: [
          { text: 'End-to-end workflow', link: '/guide/end-to-end' },
          { text: 'Authoring tasks', link: '/guide/tasks' },
          { text: 'Tuning rewards', link: '/guide/rewards' },
          { text: 'Training and tuning', link: '/guide/training' },
        ],
      },
      {
        text: 'Concepts',
        activeMatch: '^/guide/(harness|gates|semantic-correctness|context-optimization|workspace)$',
        items: [
          { text: 'The harness', link: '/guide/harness' },
          { text: 'Gates', link: '/guide/gates' },
          { text: 'Semantic correctness', link: '/guide/semantic-correctness' },
          { text: 'Context optimization', link: '/guide/context-optimization' },
          { text: 'Your workspace', link: '/guide/workspace' },
        ],
      },
      {
        text: 'Reference',
        activeMatch: '^/guide/(commands|agents|task-library)$',
        items: [
          { text: 'Commands', link: '/guide/commands' },
          { text: 'Agents', link: '/guide/agents' },
          { text: 'Task library', link: '/guide/task-library' },
        ],
      },
      { text: 'Paper', link: 'https://arxiv.org/abs/2606.08610' },
    ],

    // The sidebar mirrors the nav one-for-one, so the two never disagree about which
    // section a page belongs to.
    sidebar: {
      '/guide/': [
        {
          text: 'Getting started',
          items: [
            { text: 'What is HARBOR?', link: '/guide/' },
            { text: 'Installation', link: '/guide/install' },
            { text: 'Your first benchmark', link: '/guide/first-benchmark' },
          ],
        },
        {
          text: 'Workflows',
          items: [
            { text: 'End-to-end workflow', link: '/guide/end-to-end' },
            { text: 'Authoring tasks', link: '/guide/tasks' },
            { text: 'Tuning rewards', link: '/guide/rewards' },
            { text: 'Training and tuning', link: '/guide/training' },
          ],
        },
        {
          text: 'Concepts',
          items: [
            { text: 'The harness', link: '/guide/harness' },
            { text: 'Gates', link: '/guide/gates' },
            { text: 'Semantic correctness', link: '/guide/semantic-correctness' },
            { text: 'Context optimization', link: '/guide/context-optimization' },
            { text: 'Your workspace', link: '/guide/workspace' },
          ],
        },
        {
          text: 'Reference',
          items: [
            { text: 'Commands', link: '/guide/commands' },
            { text: 'Agents', link: '/guide/agents' },
            { text: 'Task library', link: '/guide/task-library' },
          ],
        },
      ],
    },

    socialLinks: [
      { icon: 'github', link: 'https://github.com/supersglzc/harbor-rl' },
      { icon: 'discord', link: 'https://discord.gg/W3ywA3jUKs' },
    ],

    search: { provider: 'local' },

    editLink: {
      pattern: 'https://github.com/supersglzc/harbor-rl/edit/main/docs/:path',
      text: 'Edit this page on GitHub',
    },

    footer: {
      message: 'Released under the Apache 2.0 License.',
      copyright: 'Copyright © 2026 The HARBOR Authors',
    },

    outline: [2, 3],
  },
})
