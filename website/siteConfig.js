/**
 * Copyright (c) 2017-present, Facebook, Inc.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// See https://docusaurus.io/docs/site-config.html for all the possible
// site configuration options.

const siteConfig = {
  title: 'SUMO: The Scene Understanding and Modeling Challenge' /* title for your website */,
  tagline: '',
  url: 'https://sumochallenge.org' /* your website url */,
  baseUrl: '/' /* base url for your project */,
  // cname: 'sumochallenge.org',
  // For github.io type URLs, you would set the url and baseUrl like:
  //   url: 'https://facebook.github.io',
  //   baseUrl: '/test-site/',

  // Used for publishing and more
  projectName: 'sumo-challenge',
  organizationName: 'facebookresearch',
  // For top-level user or org sites, the organization is still the same.
  // e.g., for the https://JoelMarcey.github.io site, it would be set like...
  //   organizationName: 'JoelMarcey'

  // For no header links in the top nav bar -> headerLinks: [],
  headerLinks: [],

  /* path to images for header/footer */
  headerIcon: 'img/sumo-logo-plain.png',
  favicon: 'img/sumo-logo-plain.png',

  /* colors for website */
  colors: {
    primaryColor: '#4267b2',
    secondaryColor: '#4267b2',
  },

  // This copyright info is used in /core/Footer.js and blog rss/atom feeds.
  copyright:
    'Copyright © ' +
    new Date().getFullYear() + ' ' +
    'Facebook, Inc.',

  highlight: {
    // Highlight.js theme to use for syntax highlighting in code blocks
    theme: 'default',
  },

  // Add custom scripts here that would be placed in <script> tags
  scripts: ['https://buttons.github.io/buttons.js'],

  /* On page navigation for the current documentation page */
  onPageNav: 'separate',

  /* Open Graph and Twitter card images */
  ogImage: 'img/sumo-logo-plain.png',
  twitterImage: 'img/sumo-logo-plain.png',

  // You may provide arbitrary config keys to be used as needed by your
  // template. For example, if you need your repo's URL...
  //   repoUrl: 'https://github.com/facebook/test-site',

  disableTitleTagline: true,

  // Directories inside which any css files will not be processed and concatenated to Docusaurus' styles. 
  // This is to support static html pages that may be separate from Docusaurus with completely separate styles.
  separateCss: ['static/css/bootstrap.min.css', 'static/css/da-slider.css', 'static/css/isotope.css', 'static/css/styles.css', 'static/js/owl-carousel/owl.carousel.css', 'static/js/owl-carousel/owl.carousel.css'],
};

module.exports = siteConfig;
