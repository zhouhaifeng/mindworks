var sw_anchors = {};
var urlSearchParams = new URLSearchParams(window.location.search);
var sw_timer;

/* Doc Versions */
var versions;
try {
    versions = JSON.parse(data);
}
catch (err) {
    console.log(err);
    versions = [];
}

/* Adobe Analytics */
var wapLocalCode = 'us-en';
var wapSection = 'openvinotoolkit';

(function () {
    var host = (window.document.location.protocol == 'http:') ? "http://www.intel.com" : "https://www.intel.com";
    var url = host + "/content/dam/www/global/wap/tms-loader.js"; //wap file url
    var po = document.createElement('script');
    po.type = 'text/javascript'; po.async = true; po.src = url;
    var s = document.getElementsByTagName('head')[0];
    s.appendChild(po);
})();

// legal notice for benchmarks
function addLegalNotice() {
    if (window.location.href.indexOf('openvino_docs_performance_') !== -1) {
        var legalNotice = $('<div class="opt-notice-wrapper"><p class="opt-notice">Results may vary. For workloads visit: <a href="openvino_docs_performance_benchmarks_faq.html#what-image-sizes-are-used-for-the-classification-network-models">workloads</a> and for configurations visit: <a href="openvino_docs_performance_benchmarks.html#platforms-configurations-methodology">configurations</a>. See also <a class="el" href="openvino_docs_Legal_Information.html">Legal Information</a>.</p></div>');
        $('body').append(legalNotice);
    }
}

$(document).ready(function () {
    addFooter();
    createVersions();
    updateTitleTag();
    updateLanguageSelector();
    init_col_sections();
    init_switchers();
    handleSwitcherParam();
    initViewerJS();
    addLegalNotice();
    updateSearchForm();
    initBenchmarkPickers();   // included with the new benchmarks page 
    initCollapsibleHeaders(); // included with the new benchmarks page
    createSphinxTabSets();
    initSplide();
});

// Determine where we'd go if clicking on a version selector option
function getPageUrlWithVersion(version) {
    const currentUrl = window.location.href;
    const pattern = new RegExp('(?:http|https)\:\/\/.*?\/');
    const newUrl = currentUrl.match(pattern) + version + '/index.html';
    return encodeURI(newUrl);
}

function createSphinxTabSets() {
    var sphinxTabSets = $('.sphinxtabset');
    var tabSetCount = 1000;
    sphinxTabSets.each(function() {
        var tabSet = $(this);
        var inputCount = 1;
        tabSet.addClass('tab-set docutils');
        tabSetCount++;
        tabSet.find('> .sphinxtab').each(function() {
            var tab = $(this);
            var checked = '';
            var tabValue = tab.attr('data-sphinxtab-value');
            if (inputCount == 1) {
                checked = 'checked';
            }
            var input = $(`<input ${checked} class="tab-input" id="tab-set--${tabSetCount}-input--${inputCount}" name="tab-set--${tabSetCount}" type="radio">`);
            input.insertBefore(tab);
            var label = $(`<label class="tab-label" for="tab-set--${tabSetCount}-input--${inputCount}">${tabValue}</label>`);
            label.click(onLabelClick);
            label.insertBefore(tab);
            inputCount++;
            tab.addClass('tab-content docutils');
        });

    })
    ready(); // # this function is available from tabs.js
}

function updateTitleTag() {
    var title = $('title');
    var currentVersion = getCurrentVersion();
    var newTitle = (title.text() + ' — Version(' + currentVersion + ')').replace(/\s+/g, ' ').trim();
    title.text(newTitle);
}

function getCurrentVersion() {
    var protocol = window.location.protocol + "//";
    var index = window.location.href.indexOf(protocol);
    var link = window.location.href.slice(index + protocol.length).split('/');
    var wordAfterDomain = link[1];
    if (wordAfterDomain === 'cn') {
        wordAfterDomain = link[2];
    }
    if (["index.html", "404.html", "", "latest"].indexOf(wordAfterDomain) >= 0) {
        /*
        * If this landing page, 404 or domain.com we should get first version
        * */
        return versions[0].version;
    }
    return encodeURI(wordAfterDomain);
}

function updateSearchForm() {
    var currentVersion = getCurrentVersion();
    $('.searchForm').append('<input type="hidden" name="version" value="' + currentVersion  + '">');
}

function createVersions() {
    var currentVersion = getCurrentVersion();
    var versionBtn = $('#version-selector');
    versionBtn.text(currentVersion);
    versionBtn.width((currentVersion.length * 10) + 'px');
    var versionsContainer = $('[aria-labelledby="version-selector"]');
    versions.forEach(item => {
        var link = $('<a class="dropdown-item" href="' + getPageUrlWithVersion(item.version) + '">' + item.version + '</a>');
        if (item.version === currentVersion) {
            link.addClass('font-weight-bold');
        }
        versionsContainer.append(link);
    })
    var downloadBtn = $('#download-zip-btn');
    downloadBtn.attr('href', '/archives/' + currentVersion + '.zip')
}

function updateLanguageSelector() {
    const currentVersion = getCurrentVersion();
    $('[aria-labelledby="language-selector"]').find('a').each(function(){
        const newUrl = $(this).attr('href').replace('latest', currentVersion);
        $(this).attr('href', newUrl);
    });
}

function initViewerJS() {
    try {
        var images = $('main img[src*="_images"]');
        images.each(function () {
            try {
                new Viewer($(this).get(0));
            }
            catch (err) {
                console.log(err);
            }
        });
    }
    catch (err) {
        console.log(err);
    }
}

function init_col_sections() {
    var collapsible_sections = $('div.collapsible-section');
    collapsible_sections.each(function () {
        try {
            var title = $(this).data('title') || 'Click to expand';
            var summary = $('<summary>' + title + '</summary>');
            // summary.html(title);
            var details = $('<details class="col-sect-details"></details>');
            $(this).wrap(details);
            summary.insertBefore($(this));
        }
        catch (err) {
            console.log(err);
        }
    });
}

function handleSwitcherParam() {
    var sw_type = urlSearchParams.get('sw_type');
    var section_id;
    if (sw_type && sw_type in sw_anchors) {
        section_id = sw_anchors[sw_type];
    }
    else {
        section_id = sw_anchors['default'];
    }
    $('.reference.internal.nav-link[href="#' + section_id + '"]').parent('li').css('display', 'block');
    $('#' + section_id).css('display', 'block');
    $('#button-' + section_id).removeClass('bttn-prm')
    $('#button-' + section_id).addClass('bttn-act');
    $('#button-' + section_id).attr('style', 'color: #fff !important');
}

function init_switchers() {
    var switcherAnchors = $('.switcher-anchor');
    if (switcherAnchors.length === 0) {
        return
    }
    var switcherPanel = $('<div></div>');
    switcherPanel.addClass('switcher-set');
    for (var i = 0; i < switcherAnchors.length; i++) {
        var anchor = $(switcherAnchors[i]);
        var option = $(anchor).text();
        var id = $(anchor).attr('id');
        var link = $('<a></a>');
        link.text(option);
        link.attr('href', '?sw_type=' + id);
        link.addClass('button bttn-prm button-size-m');
        switcherPanel.append(link);
        var section = $(anchor).parent('div.section');
        section.css('display', 'none');
        var section_id = section.attr('id');
        link.attr('id', 'button-' + section_id);
        $('.reference.internal.nav-link[href="#' + section_id + '"]').parent('li').css('display', 'none');
        section.addClass('switcher-content');
        sw_anchors[id] = section_id;
        if (i === 0) {
            sw_anchors['default'] = section_id;
        }
    }

    $('main').prepend(switcherPanel);
    switcherAnchors.remove();
}

// initBenchmarkPickers and initCollapsibleHeaders included with the new benchmarks page
function initBenchmarkPickers() {
    $('.picker-options .option').on('click', function(event) {
      const selectedOption = $(this).data('option');
      $('.picker-options .selectable').each(function() {
        $(this).removeClass('selected');
        const toSelect = this.classList.contains(selectedOption);
        if(toSelect) {
          $(this).addClass('selected');
        }
      });
    });
  }
  
  
  function initCollapsibleHeaders() {
    $('#performance-information-frequently-asked-questions section').on('click', function() {
      console.log($(this).find('p, table').length);
      if(!$(this).find('table, p').is(':visible')) {
        resetCollapsibleHeaders();
        $(this).find('table, p').css('display', 'block');
        $(this).find('h2').addClass('expanded')
        $(this).find('h2').get(0).scrollIntoView();
      } else {
        resetCollapsibleHeaders();
      }
    });
  
    function resetCollapsibleHeaders() {
      $('#performance-information-frequently-asked-questions section').find('h2').removeClass('expanded');
      $('#performance-information-frequently-asked-questions section p, #performance-information-frequently-asked-questions section table').hide();
    }
  }

function addFooter() {
    const footerAnchor = $('.footer');

    fetch('../footer.html').then((response) => response.text()).then((text) => {
        const footerContent = $(text);
        footerAnchor.append(footerContent);
    });
}

function initSplide() {
  const slides = $('.splide__slide');
  const height = (slides.length > 4) ? 96 + ((slides.length - 4) * 16) : 96
  var splide = new Splide('.splide', {
    direction  : 'ttb',
    type       : 'loop',
    height     : `${height}px`,
    perPage    : 1,
    autoplay   : true,
    arrows     : false,
  });
  splide.mount();
}
