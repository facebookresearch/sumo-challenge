/**
 * Copyright (c) 2017-present, Facebook, Inc.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

const React = require('react');

const CompLibrary = require('../../core/CompLibrary.js');

const Container = CompLibrary.Container;
const GridBlock = CompLibrary.GridBlock;

function Help(props) {
  const {config: siteConfig, language = ''} = props;
  const {baseUrl, docsUrl} = siteConfig;
  const docsPart = `${docsUrl ? `${docsUrl}/` : ''}`;
  const langPart = `${language ? `${language}/` : ''}`;
  const docUrl = doc => `${baseUrl}${docsPart}${langPart}${doc}`;

  const supportLinks = [
    {
      content: `[Documentation and Tutorials](${docUrl(
        'installation.html',
      )})`,
      title: 'Browse Docs',
    },
    {
      content: '[SUMO on GitHub](https://github.com/facebookresearch/sumo-challenge)',
      title: 'Join the community',
    },
    {
	content: `[Latest News](blog)`,
      title: 'Stay up to date',
    },
  ];

  return (
    <div className="docMainWrapper wrapper">
      <Container className="mainContainer documentContainer postContainer">
        <div className="post">
          <header className="postHeader">
            <h1>Need help?</h1>
          </header>
          <p>Contact Daniel Huber (dhuber@fb.com)</p>
          <GridBlock
      contents={supportLinks}
      layout="threeColumn"
	  />
        </div>
      </Container>
    </div>
  );
}

module.exports = Help;
