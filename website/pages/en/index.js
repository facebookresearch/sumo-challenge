/**
 * Copyright (c) 2017-present, Facebook, Inc.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

const React = require('react');

const CompLibrary = require('../../core/CompLibrary.js');
const MarkdownBlock = CompLibrary.MarkdownBlock; /* Used to read markdown */
const Container = CompLibrary.Container;
const GridBlock = CompLibrary.GridBlock;

const siteConfig = require(process.cwd() + '/siteConfig.js');

function imgUrl(img) {
  return siteConfig.baseUrl + 'img/' + img;
}

function docUrl(doc, language) {
  return siteConfig.baseUrl + 'docs/' + (language ? language + '/' : '') + doc;
}

function pageUrl(page, language) {
  return siteConfig.baseUrl + (language ? language + '/' : '') + page;
}

class Button extends React.Component {
  render() {
    return (
      <div className="pluginWrapper buttonWrapper">
        <a className="button" href={this.props.href} target={this.props.target}>
          {this.props.children}
        </a>
      </div>
    );
  }
}

Button.defaultProps = {
  target: '_self',
};

const SplashContainer = props => (
  <div className="homeContainer">
    <div className="homeSplashFade">
      <div className="wrapper homeWrapper">{props.children}</div>
    </div>
  </div>
);

const Logo = props => (
  <div className="projectLogo">
    <img src={props.img_src} />
  </div>
);

const ProjectTitle = props => (
  <h2 className="projectTitle">
    {siteConfig.title}
    <small>{siteConfig.tagline}</small>
  </h2>
);

const PromoSection = props => (
  <div className="section promoSection">
    <div className="promoRow">
      <div className="pluginRowBlock">{props.children}</div>
    </div>
  </div>
);

class HomeSplash extends React.Component {
  render() {
    let language = this.props.language || '';
    return (
      <SplashContainer>
        <div className="inner">
          <ProjectTitle />
          <PromoSection>
            <Button href="#try">Try It Out</Button>
            <Button href={docUrl('doc1.html', language)}>Example Link</Button>
            <Button href={docUrl('doc2.html', language)}>Example Link 2</Button>
          </PromoSection>
        </div>
      </SplashContainer>
    );
  }
}

const Block = props => (
  <Container
    padding={['bottom', 'top']}
    id={props.id}
    background={props.background}>
    <GridBlock align="center" contents={props.children} layout={props.layout} />
  </Container>
);

const Block2 = props => (
  <Container
    id={props.id}
    background={props.background}>
    <GridBlock align="left" contents={props.children} layout={props.layout} />
  </Container>
);

const Features = props => (
  <Block layout="fourColumn">
    {[
      {
        content: 'This is the content of my feature',
        image: imgUrl('docusaurus.svg'),
        imageAlign: 'top',
        title: 'Feature One',
      },
      {
        content: 'The content of my second feature',
        image: imgUrl('docusaurus.svg'),
        imageAlign: 'top',
        title: 'Feature Two',
      },
    ]}
  </Block>
);

const FeatureCallout = props => (
  <div
    className="productShowcaseSection paddingBottom"
    style={{textAlign: 'center'}}>
    <h2>Feature Callout</h2>
    <MarkdownBlock>These are features of this project</MarkdownBlock>
  </div>
);

const LearnHow = props => (
  <Block background="light">
    {[
      {
        content: 'Talk about learning how to use this',
        image: imgUrl('docusaurus.svg'),
        imageAlign: 'right',
        title: 'Learn How',
      },
    ]}
  </Block>
);

const TryOut = props => (
  <Block id="try">
    {[
      {
        content: 'Talk about trying this out',
        image: imgUrl('docusaurus.svg'),
        imageAlign: 'left',
        title: 'Try it Out',
      },
    ]}
  </Block>
);

const Description = props => (
  <Block background="dark">
    {[
      {
        content: 'This is another description of how this project is useful',
        image: imgUrl('docusaurus.svg'),
        imageAlign: 'right',
        title: 'Description',
      },
    ]}
  </Block>
);

const Showcase = props => {
  if ((siteConfig.users || []).length === 0) {
    return null;
  }
  const showcase = siteConfig.users
    .filter(user => {
      return user.pinned;
    })
    .map((user, i) => {
      return (
        <a href={user.infoLink} key={i}>
          <img src={user.image} alt={user.caption} title={user.caption} />
        </a>
      );
    });

  return (
    <div className="productShowcaseSection paddingBottom">
      <h2>{"Who's Using This?"}</h2>
      <p>This project is used by all these people</p>
      <div className="logos">{showcase}</div>
      <div className="more-users">
        <a className="button" href={pageUrl('users.html', props.language)}>
          More {siteConfig.title} Users
        </a>
      </div>
    </div>
  );
};


class Index extends React.Component {
  render() {
    let language = this.props.language || '';

      return (
<div>
    <img src={siteConfig.baseUrl + 'img/rgbd360.png'} width="100%"/>
    <div className="mainContainer">

    <Container padding={['left']}>
       The SUMO challenge aims to encourage development of
       algorithms for comprehensive scene understanding and modeling.
       The SUMO challenge task is to transform a 360 degree
       RGB-D image of an indoor scene into an instance-based 3D
       representation of that scene.  Each element in the output scene
       represents a single object, such as a wall, the floor, or a
       chair. The representation includes geometry (object shape and
       pose), appearance (color and texture), and semantics (category
       label).  The challenge includes three performance tracks, with
       elements represented in one of three increasingly descriptive
       representations: bounding boxes, voxel grids, or surface meshes.

      <h2>Latest News</h2>
	      <p><b>SUMO launch</b> (Aug 29, 2018)<br/>  Today we
	      officially launch the SUMO challenge.  We invite you to
	      participate in this exciting contest aimed at finding
	      comprehensive solutions to the problem of scene
	      understanding and modeling.   <a
	      href={siteConfig.baseUrl +
		    "blog/2018/08/29/announcement.html"}>Details</a>.
	      </p>
	  
	      <p><b>Launch delay</b> (July 30, 2018)<br/>  We are
	      experiencing some small delays in generating the data.
	      Please be patient, and we will have the data online
	      shortly. <br /> 
	      </p>

	      <p><b>SUMO Challenge Announced</b> (June 22, 2018)<br/>
	      The SUMO Challenge was announced at CVPR.  <a
	      href={siteConfig.baseUrl +
	      "blog/2018/06/22/announcement.html"}>Details</a>. 
	      </p>

       <h2>Important Dates</h2>
       <table>
       <tr> <td>June 22, 2018</td> <td>SUMO Announced at CVPR </td> </tr>
       <tr> <td>August 13, 2018</td> <td>SUMO Launch - API and data to be released</td></tr>
       <tr> <td>November 16, 2018</td> <td>Final submission deadline</td></tr>
       <tr> <td>December 3, 2018</td> <td>SUMO Workshop (ACCV 2018)</td></tr>
       </table>

       <h2>Contact Information</h2>

       For more information about the SUMO Challenge, contact Daniel Huber (dhuber at fb dot com)

     </Container>
  </div>
</div>
    );
  }
}

module.exports = Index;
