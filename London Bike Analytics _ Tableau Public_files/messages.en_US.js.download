
function getLocalizeGlobalNamespace() {
  var globalContainer;

  if (typeof process !== 'undefined' && typeof process.release !== 'undefined' && process.release.name === 'node') {
    globalContainer = global;
  } else {
    globalContainer = self;
  }

  return globalContainer;
}
!function(){function r(e){if(!(this instanceof r))return new r(e);t(e,"locale"),u(e,"locale"),this._locale=e}var a,n,i,l,t,o,u;getLocalizeGlobalNamespace().TabGlobalize=(a=function(e,r){return e=e.replace(/{[0-9a-zA-Z-_. ]+}/g,function(e){return e=e.replace(/^{([^}]*)}$/,"$1"),"string"==typeof(e=r[e])?e:"number"==typeof e?""+e:JSON.stringify(e)})},n=function(e,r,t){return r=e+(r?": "+a(r,t):""),(r=new Error(r)).code=e,function(){var t=arguments[0];[].slice.call(arguments,1).forEach(function(e){for(var r in e)t[r]=e[r]})}(r,t),r},i=function(e,r,t){e.length&&e[e.length-1].type===r?e[e.length-1].value+=t:e.push({type:r,value:t})},l=function(e,r,t,a){if(!t)throw n(e,r,a)},t=function(e,r){l("E_MISSING_PARAMETER","Missing required parameter `{name}`.",void 0!==e,{name:r})},o=function(e,r,t,a){l("E_INVALID_PAR_TYPE","Invalid `{name}` parameter ({value}). {expected} expected.",t,{expected:a,name:r,value:e})},u=function(e,r){o(e,r,void 0===e||"string"==typeof e,"a string")},r.locale=function(e){return u(e,"locale"),arguments.length&&(this._locale=e),this._locale},r._createError=n,r._formatMessage=a,r._formatMessageToParts=function(a,n){var l=0,o=[];return a.replace(/{[0-9a-zA-Z-_. ]+}/g,function(e,r){var t=e.slice(1,-1);i(o,"literal",a.slice(l,r)),i(o,"variable",n[t]),o[o.length-1].name=t,l+=r+e.length}),o.filter(function(e){return""!==e.value})},r._partsJoin=function(e){return e.map(function(e){return e.value}).join("")},r._partsPush=i,r._regexpEscape=function(e){return e.replace(/([.*+?^=!:${}()|\[\]\/\\])/g,"\\$1")},r._runtimeKey=function(e,r,t,a){return a=a||(t=t,JSON.stringify(t,function(e,r){return r&&r.runtimeKey?r.runtimeKey:r})),a=e+r+a,0<(a=[].reduce.call(a,function(e,r){return 0|(e=(e<<5)-e+r.charCodeAt(0))},0))?"a"+a:"b"+Math.abs(a)},r._stringPad=function(e,r,t){for(var a=(e="string"!=typeof e?String(e):e).length;a<r;a+=1)e=t?e+"0":"0"+e;return e},r._validateParameterPresence=t,r._validateParameterTypeString=u,r._validateParameterType=o,r)}(),function(e){var r,t,a,n;r=e.TabGlobalize,t=r._runtimeKey,a=r._validateParameterType,n=function(e,r){a(e,r,void 0===e||null!==(r=e)&&""+r=="[object Object]"||Array.isArray(e),"Array or Plain Object")},r._messageFormatterFn=function(r){return function(e){return"number"!=typeof e&&"string"!=typeof e||(e=[].slice.call(arguments,0)),n(e,"variables"),r(e)}},r._messageFormat={number:function(e,r){if(isNaN(e))throw new Error("'"+e+"' isn't a number.");return e-(r||0)},plural:function(e,r,t,a,n){if({}.hasOwnProperty.call(a,e))return a[e]();r&&(e-=r);n=t(e,n);return n in a?a[n]():a.other()},select:function(e,r){return{}.hasOwnProperty.call(r,e)?r[e]():r.other()}},r._validateParameterTypeMessageVariables=n,r.messageFormatter=r.prototype.messageFormatter=function(){return r[t("messageFormatter",this._locale,[].slice.call(arguments,0))]},r.formatMessage=r.prototype.formatMessage=function(e){return this.messageFormatter(e).apply({},[].slice.call(arguments,1))}}(getLocalizeGlobalNamespace()),function(e){var r,t,a,n,l;r=e.TabGlobalize,t=r._runtimeKey,a=r._validateParameterPresence,n=r._validateParameterType,l=function(e,r){n(e,r,void 0===e||"number"==typeof e,"Number")},r._pluralGeneratorFn=function(r){return function(e){return a(e,"value"),l(e,"value"),r(e)}},r._validateParameterTypeNumber=l,r.plural=r.prototype.plural=function(e,r){return a(e,"value"),l(e,"value"),this.pluralGenerator(r)(e)},r.pluralGenerator=r.prototype.pluralGenerator=function(e){return r[t("pluralGenerator",this._locale,[e=e||{}])]}}(getLocalizeGlobalNamespace());
(function( root, factory ) {
  root.Localize = root.Localize || {};
  root.Localize.currentMessageLocale = 'en_US';
  root.Localize.msg = new root.TabGlobalize('en');
  root.Localize.message = root.Localize.message || {};
  root.Localize.message.currentLocale = 'en_US';

  factory( root.TabGlobalize, root.Localize );
  if (root.Localize.initFormattersAndParsers) {
    root.Localize.initFormattersAndParsers();
  }
}(getLocalizeGlobalNamespace(), function( Globalize ) {
// TODO: remove workaround below
var validateParameterTypeNumber = Globalize._validateParameterTypeNumber;
var validateParameterPresence = Globalize._validateParameterPresence;
var validateParameterTypeMessageVariables = Globalize._validateParameterTypeMessageVariables;
var messageFormat = Globalize._messageFormat;
// ----
var pluralGeneratorFn = Globalize._pluralGeneratorFn;
var messageFormatterFn = Globalize._messageFormatterFn;

Globalize.a1662346136 = pluralGeneratorFn(function(n
) {
  var s = String(n).split('.'), v0 = !s[1];
  return (n == 1 && v0) ? 'one' : 'other';
});
Globalize.a1071595610 = messageFormatterFn((function(  ) {
  return function(d) { return ""; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1271652780 = messageFormatterFn((function(  ) {
  return function(d) { return "Select a trigger for the action. "; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b93358295 = messageFormatterFn((function(  ) {
  return function(d) { return "Select Sheet"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2028312755 = messageFormatterFn((function(  ) {
  return function(d) { return "Select which sheet will send data to the flow."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a479268281 = messageFormatterFn((function(  ) {
  return function(d) { return "Select sheet"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a536547316 = messageFormatterFn((function(  ) {
  return function(d) { return "Select parameters"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1453493593 = messageFormatterFn((function(  ) {
  return function(d) { return "On mark selection"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a624195482 = messageFormatterFn((function(  ) {
  return function(d) { return "Map Fields"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a599756542 = messageFormatterFn((function(  ) {
  return function(d) { return "Select which fields map to the workflow's inputs."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1581661167 = messageFormatterFn((function(  ) {
  return function(d) { return "Format Button"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1495361077 = messageFormatterFn((function(  ) {
  return function(d) { return "Select button styling."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a127999335 = messageFormatterFn((function(  ) {
  return function(d) { return "Add a constant"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b405042289 = messageFormatterFn((function(  ) {
  return function(d) { return "Clear"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b28428100 = messageFormatterFn((function(  ) {
  return function(d) { return "Required inputs "; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1275378221 = messageFormatterFn((function(  ) {
  return function(d) { return "Update the corresponding data in Salesforce Flow. Use this to enable your end users to create and run a flow directly from the dashboard."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1451152885 = messageFormatterFn((function(  ) {
  return function(d) { return "Note: You have selected to configure an External Action with Salesforce Flow. When finalized, this will share your selected data to the Salesforce org that you select. Data sent to Salesforce will be subject to your agreement with Salesforce."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1400904032 = messageFormatterFn((function(  ) {
  return function(d) { return "Browse flows in Salesforce Flow"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a458246256 = messageFormatterFn((function(  ) {
  return function(d) { return "Select a flow from Salesforce Flow. You'll configure the Tableau External Actions workflow in the next step."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b859435144 = messageFormatterFn((function(  ) {
  return function(d) { return "New Action"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2086560377 = messageFormatterFn((function(  ) {
  return function(d) { return "Configure Workflow"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1681445319 = messageFormatterFn((function(  ) {
  return function(d) { return "Next"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1333567163 = messageFormatterFn((function(  ) {
  return function(d) { return "Back"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1404076694 = messageFormatterFn((function(  ) {
  return function(d) { return "Done"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1871437902 = messageFormatterFn((function(  ) {
  return function(d) { return "Type"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b827804758 = messageFormatterFn((function(  ) {
  return function(d) { return "Created"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a865044295 = messageFormatterFn((function(  ) {
  return function(d) { return "Last updated"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b138507915 = messageFormatterFn((function(  ) {
  return function(d) { return "Send Data"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1507113380 = messageFormatterFn((function(  ) {
  return function(d) { return "Button title"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1297830569 = messageFormatterFn((function(  ) {
  return function(d) { return "Background"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b830468071 = messageFormatterFn((function(  ) {
  return function(d) { return "Border"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2036162698 = messageFormatterFn((function(  ) {
  return function(d) { return "Preview"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a685169708 = messageFormatterFn((function(  ) {
  return function(d) { return "Run workflow"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b393480156 = messageFormatterFn((function(  ) {
  return function(d) { return "Flow contains one or more unsupported data types. Go back to select a different flow, or contact your Salesforce Flow administrator to remove unsupported data types from this flow."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b891155845 = messageFormatterFn((function(  ) {
  return function(d) { return "Learn more"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1878805878 = messageFormatterFn((function(  ) {
  return function(d) { return "Add a button to the dashboard that sends user-selected data to Salesforce Flow. "; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1551936107 = messageFormatterFn((function(  ) {
  return function(d) { return "No results found"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b271370890 = messageFormatterFn((function(  ) {
  return function(d) { return "Try a different query or create a new action. "; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a809066878 = messageFormatterFn((function(  ) {
  return function(d) { return "Configure \"" + d.ACTION_NAME + "\""; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b572482778 = messageFormatterFn((function(  ) {
  return function(d) { return "Trigger: Button selection"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a831507873 = messageFormatterFn((function(  ) {
  return function(d) { return "Trigger: Mark selection in \"" + d.WORKSHEET_NAME + "\""; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1933944357 = messageFormatterFn((function(  ) {
  return function(d) { return "Trigger: Parameter change"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1990748451 = messageFormatterFn((function(  ) {
  return function(d) { return "Refresh"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a106925255 = messageFormatterFn((function(  ) {
  return function(d) { return "By " + d.CREATOR; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1671354798 = messageFormatterFn((function(  ) {
  return function(d) { return "Undefined"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b401072035 = messageFormatterFn((function(  ) {
  return function(d) { return "Action isn't configured"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1633253650 = messageFormatterFn((function(  ) {
  return function(d) { return "Search flows"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a153638163 = messageFormatterFn((function(  ) {
  return function(d) { return d.DATA_TYPE + " is an unsupported data type. "; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1675372905 = messageFormatterFn((function(  ) {
  return function(d) { return "Dashboard Properties"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a722690911 = messageFormatterFn((function(  ) {
  return function(d) { return "Columns"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1987436541 = messageFormatterFn((function(  ) {
  return function(d) { return "Filters"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1063088771 = messageFormatterFn((function(  ) {
  return function(d) { return "Filters(" + d.SHEET_NAME + ")"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1383379006 = messageFormatterFn((function(  ) {
  return function(d) { return "Parameters"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a69611950 = messageFormatterFn((function(  ) {
  return function(d) { return "Workflow completed"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a399287278 = messageFormatterFn((function(  ) {
  return function(d) { return "Error occurred: " + d.EXCEPTION_CODE + ". Contact your Salesforce Flow administrator."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b232199902 = messageFormatterFn((function(  ) {
  return function(d) { return "Workflow failed. Contact your Salesforce Flow administrator."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1750807011 = messageFormatterFn((function(  ) {
  return function(d) { return "Unable to submit due to workflow configuration error. Contact your dashboard author."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1797916570 = messageFormatterFn((function(  ) {
  return function(d) { return "Unable to submit data to Salesforce Flow."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2029674486 = messageFormatterFn((function(  ) {
  return function(d) { return "Unable to submit data to flow. Contact your workflow author."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1070340919 = messageFormatterFn((function(  ) {
  return function(d) { return "Error fetching flow inputs"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1646636889 = messageFormatterFn((function(  ) {
  return function(d) { return "One or more data type mismatches exist in the mapping of input fields."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1962042431 = messageFormatterFn((function(  ) {
  return function(d) { return "No selected marks found."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a223436063 = messageFormatterFn((function(  ) {
  return function(d) { return "Sheet for workflow removed. Undo deletion or have dashboard author configure a new sheet for the workflow."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a984846435 = messageFormatterFn((function(  ) {
  return function(d) { return "Please login to Salesforce"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b987502964 = messageFormatterFn((function(  ) {
  return function(d) { return "Login to Salesforce"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a255377002 = messageFormatterFn((function(  ) {
  return function(d) { return "Action was triggered but may take some time to complete."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1622153453 = messageFormatterFn((function(  ) {
  return function(d) { return "Workflow"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1760238794 = messageFormatterFn((function(  ) {
  return function(d) { return "No sheets available. Add a sheet to the dashboard, and retry."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a302966253 = messageFormatterFn((function(  ) {
  return function(d) { return "Set Mark Selection Limit"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1724599535 = messageFormatterFn((function(  ) {
  return function(d) { return "Limit the number of marks that users can select from the dashboard and send to the flow. (optional)"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1721419172 = messageFormatterFn((function(  ) {
  return function(d) { return "No limit set"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2060475864 = messageFormatterFn((function(  ) {
  return function(d) { return "Mark limit = " + d.MAX_MARKS; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a695115662 = messageFormatterFn((function(  ) {
  return function(d) { return "Must be a non-zero, positive integer or blank."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b953511190 = messageFormatterFn((function(  ) {
  return function(d) { return "The selected flow only supports single values."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b841043319 = messageFormatterFn((function(  ) {
  return function(d) { return "Select a maximum of " + d.MAX_MARKS + " marks"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1335739514 = messageFormatterFn((function(  ) {
  return function(d) { return "Select a maximum of 1 mark"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b996336404 = messageFormatterFn((function(  ) {
  return function(d) { return "Only a single value from the array will be sent to the flow."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a34147230 = messageFormatterFn((function(  ) {
  return function(d) { return "Single values"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1834031071 = messageFormatterFn((function(  ) {
  return function(d) { return "Accepts multiple values"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b272959128 = messageFormatterFn((function(  ) {
  return function(d) { return "Marks must be selected from only one pane."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2130533634 = messageFormatterFn((function(  ) {
  return function(d) { return "Uh oh! You need to be authenticated to use this extension"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1736479392 = messageFormatterFn((function(  ) {
  return function(d) { return "Select optional field to map"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b214784595 = messageFormatterFn((function(  ) {
  return function(d) { return "Add field"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2021119454 = messageFormatterFn((function(  ) {
  return function(d) { return "Address Geocoding"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b887763298 = messageFormatterFn((function(  ) {
  return function(d) { return "Choose a value"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2004572503 = messageFormatterFn((function(  ) {
  return function(d) { return "Cancel"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1168827627 = messageFormatterFn((function(  ) {
  return function(d) { return "OK"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1061946458 = messageFormatterFn((function(  ) {
  return function(d) { return "Cross-Database Join"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1020264557 = messageFormatterFn((function(  ) {
  return function(d) { return "Choose where the join happens when joining data from multiple sources."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1164214669 = messageFormatterFn((function(  ) {
  return function(d) { return "The workbook uses cross-database joins that are always performed in the database. This option doesn't consider file size when moving the file-based data into the database."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1213959207 = messageFormatterFn((function(  ) {
  return function(d) { return "Learn more"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1669508057 = messageFormatterFn((function(  ) {
  return function(d) { return "OK"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2133846203 = messageFormatterFn((function(  ) {
  return function(d) { return "Cancel"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1959906570 = messageFormatterFn((function(  ) {
  return function(d) { return "Select one of the following to continue:"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b817799712 = messageFormatterFn((function(  ) {
  return function(d) { return "Always perform joins in the database"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1664131323 = messageFormatterFn((function(  ) {
  return function(d) { return "Data is moved from a file-based connection to the database. This option ignores the file's size and may impact performance."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b564084659 = messageFormatterFn((function(  ) {
  return function(d) { return "Use this option only if the database is from a trusted source"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1812583788 = messageFormatterFn((function(  ) {
  return function(d) { return "Let Tableau decide where to join (default)"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1281567559 = messageFormatterFn((function(  ) {
  return function(d) { return "Data may be moved across connections and joined in a database, or the join may occur in Tableau."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a871988918 = messageFormatterFn((function(  ) {
  return function(d) { return "Alternatively, let Tableau decide whether to perform the join locally or move data into the database to perform the join there. Letting Tableau decide may improve performance."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1405732433 = messageFormatterFn((function(  ) {
  return function(d) { return "Show details"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b515312374 = messageFormatterFn((function(  ) {
  return function(d) { return "Hide details"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a229986360 = messageFormatterFn((function(  ) {
  return function(d) { return "Datasource"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1871492545 = messageFormatterFn((function(  ) {
  return function(d) { return "Database Connection"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2067226219 = messageFormatterFn((function(  ) {
  return function(d) { return "File Connection"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1019335495 = messageFormatterFn((function(  ) {
  return function(d) { return "Let Tableau Decide"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1542092929 = messageFormatterFn((function(  ) {
  return function(d) { return "Keep Using the Database"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b210572163 = messageFormatterFn((function(  ) {
  return function(d) { return "You must save the workbook before configuring an analytics extension."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2006962331 = messageFormatterFn((function(  ) {
  return function(d) { return "Please save workbook"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a89292959 = messageFormatterFn((function(  ) {
  return function(d) { return "I applied your filter but all the data has been filtered out. Try updating your filters or remove them to see results."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b6251466 = messageFormatterFn((function(  ) {
  return function(d) { return "and"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a325421752 = messageFormatterFn((function(  ) {
  return function(d) { return "OK, the new " + d.field + " field was added to the Data pane."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a273024389 = messageFormatterFn((function(  ) {
  return function(d) { return "OK, the updated " + d.field + " field was added to the Data pane."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1428796247 = messageFormatterFn((function(  ) {
  return function(d) { return "I can't create a viz based on your request."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1222587535 = messageFormatterFn((function(  ) {
  return function(d) { return "Learn about tasks I can help with."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b602042945 = messageFormatterFn((function(  ) {
  return function(d) { return "This tool uses generative AI, which can produce inaccurate or harmful responses. Review for accuracy and safety before using."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a660763968 = messageFormatterFn((function(  ) {
  return function(d) { return "Edit"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1955158194 = messageFormatterFn((function(  ) {
  return function(d) { return "Describe your task or ask a question and let Tableau Agent create a viz or calculation for you."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b269052544 = messageFormatterFn((function(  ) {
  return function(d) { return "Thanks for your feedback!"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b590912165 = messageFormatterFn((function(  ) {
  return function(d) { return "Your feedback can't be sent right now. Try again later."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a202103872 = messageFormatterFn((function(  ) {
  return function(d) { return "Indexing complete."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a104815814 = messageFormatterFn((function(  ) {
  return function(d) { return "I couldn't describe that calc. Check the field name and try asking me again."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b769908072 = messageFormatterFn((function(  ) {
  return function(d) { return "Learn more"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1764322853 = messageFormatterFn((function(  ) {
  return function(d) { return "AI in Tableau is temporarily unavailable. Try again later."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1838345061 = messageFormatterFn((function(  ) {
  return function(d) { return "Try phrasing your request another way."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1939482861 = messageFormatterFn((function(  ) {
  return function(d) { return "OK"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1029335902 = messageFormatterFn((function(  ) {
  return function(d) { return "Tableau Agent"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1446279515 = messageFormatterFn((function(  ) {
  return function(d) { return "Recreate"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1888537792 = messageFormatterFn((function(  ) {
  return function(d) { return "Retry with Tableau Agent"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1172265230 = messageFormatterFn((function(  ) {
  return function(d) { return "Looks like something's wrong with our connection. Try again."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a595183245 = messageFormatterFn((function(  ) {
  return function(d) { return "It looks like you switched to a new data source. Switch back to the original data source to add a filter to the viz. Otherwise, tell me what you'd like to do."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a497911806 = messageFormatterFn((function(  ) {
  return function(d) { return "It looks like you switched to a new data source. Switch back to the original data source to recreate the viz. Otherwise, tell me what you'd like to do."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a921120884 = messageFormatterFn((function(  ) {
  return function(d) { return "It looks like you switched to a new data source. Switch back to the original data source to retry the calc. Otherwise, tell me what you'd like to do."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1457243690 = messageFormatterFn((function(  ) {
  return function(d) { return "It looks like you switched to a new data source. Switch back to the original data source to retry your request. Otherwise, tell me what you'd like to do."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b895784082 = messageFormatterFn((function(  ) {
  return function(d) { return "It looks like you switched to a new data source. Switch back to the original data source to retry the viz. Otherwise, tell me what you'd like to do."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b809169210 = messageFormatterFn((function(  ) {
  return function(d) { return "It looks like you switched to a new data source since I made that suggestion. Switch back to the original data source, and then try again."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1418586872 = messageFormatterFn((function(  ) {
  return function(d) { return "Something went wrong. Try again."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a116326522 = messageFormatterFn((function(  ) {
  return function(d) { return "Tableau Agent requires a newer version of Tableau Desktop. Upgrade Tableau Desktop to the latest version."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b240371317 = messageFormatterFn((function(  ) {
  return function(d) { return "Describe your task or ask a question..."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1773278550 = messageFormatterFn((function(  ) {
  return function(d) { return "Building your viz"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b154303869 = messageFormatterFn((function(  ) {
  return function(d) { return "Indexing your data\nThis may take a minute"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2101498863 = messageFormatterFn((function(  ) {
  return function(d) { return "Working on it"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1547321680 = messageFormatterFn((function(  ) {
  return function(d) { return "You're about to use Tableau Agent"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1262011282 = messageFormatterFn((function(  ) {
  return function(d) { return "Tableau Agent is a conversational AI assistant to help you explore and analyze your data. Describe your task or ask a question and let Tableau Agent create a viz for you."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a294238547 = messageFormatterFn((function(  ) {
  return function(d) { return "Tableau Agent is a conversational AI assistant to help you analyze your data and find insights."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a322704075 = messageFormatterFn((function(  ) {
  return function(d) { return "Features powered by Generative AI can produce inaccurate or harmful responses. Review the output for accuracy and safety. You assume responsibility for how the outcomes of Agents and other AI features are applied to your organization."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b519469367 = messageFormatterFn((function(  ) {
  return function(d) { return "Got It"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1093165395 = messageFormatterFn((function(  ) {
  return function(d) { return "Hi, I'm Tableau Agent, a conversational AI assistant. I can help you explore and analyze your data. To get started, you'll need to sign in to a Tableau Cloud site that has Tableau+ and AI in Tableau turned on, then switch to a sheet tab."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b918627176 = messageFormatterFn((function(  ) {
  return function(d) { return "Tableau Agent is only available for worksheets. Switch to a sheet tab to try Tableau Agent."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a543166609 = messageFormatterFn((function(  ) {
  return function(d) { return "Tableau Agent works with your data. Connect to a data source to use Tableau Agent."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1397970498 = messageFormatterFn((function(  ) {
  return function(d) { return "What can I help you with?"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b838207498 = messageFormatterFn((function(  ) {
  return function(d) { return "You can ask me questions or describe a task you want to do in Tableau. For example, I can help you:"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1105952402 = messageFormatterFn((function(  ) {
  return function(d) { return "Build a viz"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1105953363 = messageFormatterFn((function(  ) {
  return function(d) { return "Filter and sort data"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1105954324 = messageFormatterFn((function(  ) {
  return function(d) { return "Create a calculated field"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1404947779 = messageFormatterFn((function(  ) {
  return function(d) { return "Hi, I'm Tableau Agent, your AI assistant. I can look at your viz or dashboard and highlight insights about the data."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b531806534 = messageFormatterFn((function(  ) {
  return function(d) { return "Hi, I'm Tableau Agent, an AI assistant. I can help draft a formula for you in Tableau syntax."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1730922430 = messageFormatterFn((function(  ) {
  return function(d) { return "Right now I can help with:"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a428375613 = messageFormatterFn((function(  ) {
  return function(d) { return "If-then scenarios (logical functions)"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a428376574 = messageFormatterFn((function(  ) {
  return function(d) { return "Mathematical operations (numerical functions)"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a428377535 = messageFormatterFn((function(  ) {
  return function(d) { return "Manipulating text fields like extracting parts of a field value (string functions)"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a428378496 = messageFormatterFn((function(  ) {
  return function(d) { return "Explain the calculation"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1718010643 = messageFormatterFn((function(  ) {
  return function(d) { return "Just describe what you want to do and I'll get the calculation started."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a876131157 = messageFormatterFn((function(  ) {
  return function(d) { return "What calculation can I help you with?"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a968126704 = messageFormatterFn((function(  ) {
  return function(d) { return "OK"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b494140193 = messageFormatterFn((function(  ) {
  return function(d) { return "I can't find '" + d.field + "' in your data set. Is there a different field I should use? Select one of the options below or enter more details in the text box."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1790119185 = messageFormatterFn((function(  ) {
  return function(d) { return "Add Filter"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1971331629 = messageFormatterFn((function(  ) {
  return function(d) { return "To make this viz, the " + d.field + " field needs to be filtered to " + d.fieldValues + ". To make sure the right values are selected, I'll need your help:"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1936863810 = messageFormatterFn((function(  ) {
  return function(d) { return "Your requested viz needs filters on " + d.field1 + " and " + d.field2 + ". To make sure the right values are selected, I'll need your help:"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1679842546 = messageFormatterFn((function(  ) {
  return function(d) { return "Your requested viz needs filters on " + d.fields + ". To make sure the right values are selected, I'll need your help:"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1583558598 = messageFormatterFn((function(  ) {
  return function(d) { return "1. Select the Add Filter button and I'll create a filter on " + d.field; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1299465031 = messageFormatterFn((function(  ) {
  return function(d) { return "1. Select the Add Filter button and I'll create the first filter on " + d.field; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b689504100 = messageFormatterFn((function(  ) {
  return function(d) { return "2. Specify the values you want to keep"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b689503139 = messageFormatterFn((function(  ) {
  return function(d) { return "3. Select OK to create the filter"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a197128006 = messageFormatterFn((function(  ) {
  return function(d) { return "4. Repeat for the second filter on " + d.field; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b245361947 = messageFormatterFn((function(  ) {
  return function(d) { return "4. Repeat for the remaining filters"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b993730170 = messageFormatterFn((function(  ) {
  return function(d) { return "OK, I've updated the viz to show only the values you selected."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1248394583 = messageFormatterFn((function(  ) {
  return function(d) { return "I've updated the viz to show no values. When all values are filtered out, the viz will be blank."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1581765908 = messageFormatterFn((function(  ) {
  return function(d) { return "OK. Here's a viz based on:"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1581766869 = messageFormatterFn((function(  ) {
  return function(d) { return "Got it. Here's a viz based on:"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1581767830 = messageFormatterFn((function(  ) {
  return function(d) { return "Alright. Here's a viz based on:"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1657400751 = messageFormatterFn((function(  ) {
  return function(d) { return "OK. I can't make exactly what you requested, but here is " + d.chart + " instead."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1657401712 = messageFormatterFn((function(  ) {
  return function(d) { return "Got it. I can't make exactly what you requested, but here is " + d.chart + " instead."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1657402673 = messageFormatterFn((function(  ) {
  return function(d) { return "Alright. I can't make exactly what you requested, but here is " + d.chart + " instead."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a147630784 = messageFormatterFn((function(  ) {
  return function(d) { return "OK. I can't make exactly what you requested, but here's a viz based on the fields you mentioned."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a147631745 = messageFormatterFn((function(  ) {
  return function(d) { return "Got it. I can't make exactly what you requested, but here's a viz based on the fields you mentioned."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a147632706 = messageFormatterFn((function(  ) {
  return function(d) { return "Alright. I can't make exactly what you requested, but here's a viz based on the fields you mentioned."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b439623792 = messageFormatterFn((function(  ) {
  return function(d) { return "a text table"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1245900119 = messageFormatterFn((function(  ) {
  return function(d) { return "a heatmap"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a810832753 = messageFormatterFn((function(  ) {
  return function(d) { return "a stacked bar chart"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1504066594 = messageFormatterFn((function(  ) {
  return function(d) { return "a bar chart"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a787355369 = messageFormatterFn((function(  ) {
  return function(d) { return "a line chart"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b189616435 = messageFormatterFn((function(  ) {
  return function(d) { return "a dual-line chart"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1983723472 = messageFormatterFn((function(  ) {
  return function(d) { return "an area chart"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b653320767 = messageFormatterFn((function(  ) {
  return function(d) { return "a gantt chart"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1639194090 = messageFormatterFn((function(  ) {
  return function(d) { return "a scatterplot"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a462664881 = messageFormatterFn((function(  ) {
  return function(d) { return "a histogram"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a151764665 = messageFormatterFn((function(  ) {
  return function(d) { return "a map"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b836110633 = messageFormatterFn((function(  ) {
  return function(d) { return "a treemap chart"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a291413129 = messageFormatterFn((function(  ) {
  return function(d) { return "a pie chart"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b536543511 = messageFormatterFn((function(  ) {
  return function(d) { return "a boxplot"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a98868507 = messageFormatterFn((function(  ) {
  return function(d) { return "a bullet chart"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1580949935 = messageFormatterFn((function(  ) {
  return function(d) { return "a bubble chart"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1238707254 = messageFormatterFn((function(  ) {
  return function(d) { return "It looks like you switched sheets while Tableau Agent was still processing. Switch back to the original sheet and type your request again to continue. Otherwise, tell me what you'd like to do."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1054415238 = messageFormatterFn((function(  ) {
  return function(d) { return "Suggestions"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1435558869 = messageFormatterFn((function(  ) {
  return function(d) { return "Select an option below or click the refresh button to see more options."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1772092372 = messageFormatterFn((function(  ) {
  return function(d) { return "No suggested questions. Try describing your request in the text box below."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1930872637 = messageFormatterFn((function(  ) {
  return function(d) { return "I'm having trouble with that."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a185241440 = messageFormatterFn((function(  ) {
  return function(d) { return "Try phrasing your request another way, or click Suggestions to see what else you can try."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b86840862 = messageFormatterFn((function(  ) {
  return function(d) { return "Get Insights"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b194768500 = messageFormatterFn((function(  ) {
  return function(d) { return "Your system is set to " + d.language + ", which Tableau Agent doesn't currently support. Responses might be in English. "; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1343988844 = messageFormatterFn((function(  ) {
  return function(d) { return "Your system is set to a language that Tableau Agent doesn't currently support. Responses might be in English. "; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1805807501 = messageFormatterFn((function(  ) {
  return function(d) { return "Failed to copy content to clipboard:"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1403215793 = messageFormatterFn((function(  ) {
  return function(d) { return "Invalid JSON format: Expected an object. Response from endpoints API: " + d.rawResponse; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1816634163 = messageFormatterFn((function(  ) {
  return function(d) { return "Invalid deployed function data for key " + d.key + ". Response from endpoints API: " + d.rawResponse; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1870341883 = messageFormatterFn((function(  ) {
  return function(d) { return "There are currently no custom functions deployed on the Analytics Extension. Reach out to your server administrator to add functions and refer to Tabpy wiki on how to add custom functions: https://tableau.github.io/TabPy/docs/tabpy-tools.html. Response from endpoints API: " + d.rawResponse; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1205596567 = messageFormatterFn((function(  ) {
  return function(d) { return "No custom function selected"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2054417637 = messageFormatterFn((function(  ) {
  return function(d) { return "Please select a custom function from the display to see more details"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a762246666 = messageFormatterFn((function(  ) {
  return function(d) { return "Server: " + d.host + ":" + d.port; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a687231626 = messageFormatterFn((function(  ) {
  return function(d) { return "Custom Functions Explorer"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1490963944 = messageFormatterFn((function(  ) {
  return function(d) { return "Search and explore your organizations custom functions below and add them to your workbook. These python functions are managed by your company and deployed to the " + d.host + " server."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1945141836 = messageFormatterFn((function(  ) {
  return function(d) { return "Search custom functions"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1197835837 = messageFormatterFn((function(  ) {
  return function(d) { return "No functions available"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1619240277 = messageFormatterFn((function(  ) {
  return function(d) { return "Contact your administrator to add custom functions"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1687223154 = messageFormatterFn((function(  ) {
  return function(d) { return "required"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1081996025 = messageFormatterFn((function(  ) {
  return function(d) { return "Copy to Clipboard"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b353645046 = messageFormatterFn((function(  ) {
  return function(d) { return "Copied to Clipboard"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a658439443 = messageFormatterFn((function(  ) {
  return function(d) { return "Copy syntax to clipboard"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b544686878 = messageFormatterFn((function(  ) {
  return function(d) { return "Copied"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1213850276 = messageFormatterFn((function(  ) {
  return function(d) { return "No input schema provided"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1496802015 = messageFormatterFn((function(  ) {
  return function(d) { return "No output schema provided"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1814066981 = messageFormatterFn((function(  ) {
  return function(d) { return "Inputs"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1017686771 = messageFormatterFn((function(  ) {
  return function(d) { return "Output"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a572327806 = messageFormatterFn((function(  ) {
  return function(d) { return "Outputs"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b98590741 = messageFormatterFn((function(  ) {
  return function(d) { return "Syntax"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b708690379 = messageFormatterFn((function(  ) {
  return function(d) { return "Updated"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1166467404 = messageFormatterFn((function(  ) {
  return function(d) { return "Learn more"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1757141796 = messageFormatterFn((function(  ) {
  return function(d) { return "Data type: number"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a118089631 = messageFormatterFn((function(  ) {
  return function(d) { return "Data type: integer"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a301760788 = messageFormatterFn((function(  ) {
  return function(d) { return "Data type: string"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a504144885 = messageFormatterFn((function(  ) {
  return function(d) { return "Data type: boolean"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b925399556 = messageFormatterFn((function(  ) {
  return function(d) { return "Data type: default"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2134615749 = messageFormatterFn((function(  ) {
  return function(d) { return "Search Custom Functions"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a77549064 = messageFormatterFn((function(  ) {
  return function(d) { return "Animation"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b67870274 = messageFormatterFn((function(  ) {
  return function(d) { return "Change account settings"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1279425506 = messageFormatterFn((function(  ) {
  return function(d) { return "Close"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a801045609 = messageFormatterFn((function(  ) {
  return function(d) { return "Custom"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a243949919 = messageFormatterFn((function(  ) {
  return function(d) { return "Default"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1683490262 = messageFormatterFn((function(  ) {
  return function(d) { return "Workbook Default"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a670725070 = messageFormatterFn((function(  ) {
  return function(d) { return "Mark Transitions"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2069135199 = messageFormatterFn((function(  ) {
  return function(d) { return "Animations are currently disabled. You can enable them in the Help > Settings and Performance menu."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1993222959 = messageFormatterFn((function(  ) {
  return function(d) { return "Animations are disabled for your site. For more information, contact your Tableau administrator."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a422768827 = messageFormatterFn((function(  ) {
  return function(d) { return "Animations are disabled for your user account."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b657202522 = messageFormatterFn((function(  ) {
  return function(d) { return d["0"] + " seconds (" + d["1"] + ")"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1350651614 = messageFormatterFn((function(  ) {
  return function(d) { return "Duration"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b856357143 = messageFormatterFn((function(  ) {
  return function(d) { return "Format Animations"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a889788793 = messageFormatterFn((function(  ) {
  return function(d) { return d["0"] + " (Default)"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a138015997 = messageFormatterFn((function(  ) {
  return function(d) { return "Duration can't be negative"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2134238379 = messageFormatterFn((function(  ) {
  return function(d) { return "(None Selected)"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b418740643 = messageFormatterFn((function(  ) {
  return function(d) { return "Must be a number"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1654805377 = messageFormatterFn((function(  ) {
  return function(d) { return "Off"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b152656053 = messageFormatterFn((function(  ) {
  return function(d) { return "On"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b42811684 = messageFormatterFn((function(  ) {
  return function(d) { return "Reset All Sheets"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a928497181 = messageFormatterFn((function(  ) {
  return function(d) { return "sec"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1865087693 = messageFormatterFn((function(  ) {
  return function(d) { return "Selected Sheet"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1861746697 = messageFormatterFn((function(  ) {
  return function(d) { return "Sequential"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1948867789 = messageFormatterFn((function(  ) {
  return function(d) { return "Simultaneous"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1896786501 = messageFormatterFn((function(  ) {
  return function(d) { return "Style"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1780016105 = messageFormatterFn((function(  ) {
  return function(d) { return "Learn more"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a232749189 = messageFormatterFn((function(  ) {
  return function(d) { return "Some sheets might not animate when published."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1188444745 = messageFormatterFn((function(  ) {
  return function(d) { return "Some animations in this workbook play only in Tableau Desktop."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1579575489 = messageFormatterFn((function(  ) {
  return function(d) { return "All animations are compatible with Tableau Desktop, but some might not play when published to Tableau Server."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b753144018 = messageFormatterFn((function(  ) {
  return function(d) { return "Show template project contributors!"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1362160461 = messageFormatterFn((function(  ) {
  return function(d) { return "Hide template project contributors!"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b48845964 = messageFormatterFn((function(  ) {
  return function(d) { return "Ask Data"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1661245757 = messageFormatterFn((function(  ) {
  return function(d) { return "Ask Data lens is unavailable"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a168410757 = messageFormatterFn((function(  ) {
  return function(d) { return "Ask Data is disabled on this site"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a332555265 = messageFormatterFn((function(  ) {
  return function(d) { return "OK"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1954686913 = messageFormatterFn((function(  ) {
  return function(d) { return "Cancel"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1752344273 = messageFormatterFn((function(  ) {
  return function(d) { return "New Lens"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a58214691 = messageFormatterFn((function(  ) {
  return function(d) { return "Select an Ask Data lens and set its optional features."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a66646399 = messageFormatterFn((function(  ) {
  return function(d) { return "Configure Ask Data Object"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b745000971 = messageFormatterFn((function(  ) {
  return function(d) { return "Open Data Source"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a571985761 = messageFormatterFn((function(  ) {
  return function(d) { return "To configure Ask Data, edit this dashboard in Tableau Cloud or Tableau Server."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1157771270 = messageFormatterFn((function(  ) {
  return function(d) { return "From the Server menu, select Sign In to use Ask Data"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a765382109 = messageFormatterFn((function(  ) {
  return function(d) { return "Can’t configure Ask Data until this workbook is connected to a published data source"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1635266971 = messageFormatterFn((function(  ) {
  return function(d) { return "Couldn't refresh data sources."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b458481734 = messageFormatterFn((function(  ) {
  return function(d) { return "This version of the Ask Data Dashboard Object (" + d.version + ") isn’t supported by your server"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1046371696 = messageFormatterFn((function(  ) {
  return function(d) { return "Ask Data isn't configured"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b121183374 = messageFormatterFn((function(  ) {
  return function(d) { return "Ask Data isn't supported by this version of Tableau Server"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a319414479 = messageFormatterFn((function(  ) {
  return function(d) { return "Can't load Ask Data"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1862905051 = messageFormatterFn((function(  ) {
  return function(d) { return "Can't load lenses"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1949847003 = messageFormatterFn((function(  ) {
  return function(d) { return "The server isn't responding"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1254382266 = messageFormatterFn((function(  ) {
  return function(d) { return "Unable to authenticate. Refresh the dashboard to load Ask Data."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1835551199 = messageFormatterFn((function(  ) {
  return function(d) { return "Couldn't create a new Ask Data lens."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b483312338 = messageFormatterFn((function(  ) {
  return function(d) { return "You don't have permission to create an Ask Data lens."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1005006703 = messageFormatterFn((function(  ) {
  return function(d) { return "Failed to apply selected lens."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1795714469 = messageFormatterFn((function(  ) {
  return function(d) { return "For more information, contact your Tableau adminstrator."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2095243502 = messageFormatterFn((function(  ) {
  return function(d) { return "Learn more"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a680387608 = messageFormatterFn((function(  ) {
  return function(d) { return "Name"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1135523036 = messageFormatterFn((function(  ) {
  return function(d) { return "Owner"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1216964330 = messageFormatterFn((function(  ) {
  return function(d) { return "Modified on"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2037915605 = messageFormatterFn((function(  ) {
  return function(d) { return "Data Source"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a423373679 = messageFormatterFn((function(  ) {
  return function(d) { return "Ask Data settings such as synonyms, selected fields, and recommended visualizations are published as a lens."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1507431993 = messageFormatterFn((function(  ) {
  return function(d) { return "Refresh List"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a678185027 = messageFormatterFn((function(  ) {
  return function(d) { return "Ask Data Lens"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a31306160 = messageFormatterFn((function(  ) {
  return function(d) { return "Hidden User"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1580335381 = messageFormatterFn((function(  ) {
  return function(d) { return "No connected data sources"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b556320496 = messageFormatterFn((function(  ) {
  return function(d) { return "No published data sources"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1953701846 = messageFormatterFn((function(  ) {
  return function(d) { return "To use Ask Data, first connect this workbook to a published data source."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b252927761 = messageFormatterFn((function(  ) {
  return function(d) { return "Refresh data sources"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1288356616 = messageFormatterFn((function(  ) {
  return function(d) { return "Refreshing..."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b203970058 = messageFormatterFn((function(  ) {
  return function(d) { return "No lenses for this data source"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a645041242 = messageFormatterFn((function(  ) {
  return function(d) { return "Go to the data source page and create a new lens. Then click Refresh above."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1328567402 = messageFormatterFn((function(  ) {
  return function(d) { return "Go to Data Source Page"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1747530523 = messageFormatterFn((function(  ) {
  return function(d) { return "Publish workbook to use Ask Data"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1092164022 = messageFormatterFn((function(  ) {
  return function(d) { return "Ask Data needs a workbook published to Tableau Cloud or Tableau Server and connected to a published data source there."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1817702615 = messageFormatterFn((function(  ) {
  return function(d) { return "Select Another Lens"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a934253888 = messageFormatterFn((function(  ) {
  return function(d) { return "The lens or its data source has been removed from the site, or you no longer have permission to access them."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1197402014 = messageFormatterFn((function(  ) {
  return function(d) { return "Lens Toolbar Options"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a925857215 = messageFormatterFn((function(  ) {
  return function(d) { return "Add Visualization to Pins"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1284397143 = messageFormatterFn((function(  ) {
  return function(d) { return "Publish as Workbook"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b980145719 = messageFormatterFn((function(  ) {
  return function(d) { return "Share Visualization"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1405900355 = messageFormatterFn((function(  ) {
  return function(d) { return "Try again"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a435779359 = messageFormatterFn((function(  ) {
  return function(d) { return "Try again later"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a605106846 = messageFormatterFn((function(  ) {
  return function(d) { return "Show template project contributors!"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b3909597 = messageFormatterFn((function(  ) {
  return function(d) { return "Hide template project contributors!"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a671904401 = messageFormatterFn((function(  ) {
  return function(d) { return "Extend Date Range"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2039400223 = messageFormatterFn((function(  ) {
  return function(d) { return "Extend the date range to show values for forecasted and calculated fields."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b100791223 = messageFormatterFn((function(  ) {
  return function(d) { return "Length of additional time"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b344725493 = messageFormatterFn((function(  ) {
  return function(d) { return "Length of time"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a289988440 = messageFormatterFn((function(  ) {
  return function(d) { return "Unit of time"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a414229403 = messageFormatterFn((function(  ) {
  return function(d) { return "YEAR"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a338109220 = messageFormatterFn((function(  ) {
  return function(d) { return "QUARTER"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1805477576 = messageFormatterFn((function(  ) {
  return function(d) { return "MONTH"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a357083538 = messageFormatterFn((function(  ) {
  return function(d) { return "WEEK"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1945792876 = messageFormatterFn((function(  ) {
  return function(d) { return "DAY"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b62635134 = messageFormatterFn((function(  ) {
  return function(d) { return "HOUR"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1163941134 = messageFormatterFn((function(  ) {
  return function(d) { return "MINUTE"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1127297362 = messageFormatterFn((function(  ) {
  return function(d) { return "SECOND"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b375773432 = messageFormatterFn((function(  ) {
  return function(d) { return "Image"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2075137611 = messageFormatterFn((function(  ) {
  return function(d) { return "Choose"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b314353881 = messageFormatterFn((function(  ) {
  return function(d) { return "Choose an image..."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b35925874 = messageFormatterFn((function(  ) {
  return function(d) { return "Tooltip"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1167768531 = messageFormatterFn((function(  ) {
  return function(d) { return "Enter optional tooltip text"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b825278119 = messageFormatterFn((function(  ) {
  return function(d) { return "OK"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a124087047 = messageFormatterFn((function(  ) {
  return function(d) { return "Apply"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1483567957 = messageFormatterFn((function(  ) {
  return function(d) { return "None"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a456387273 = messageFormatterFn((function(  ) {
  return function(d) { return "Border"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a603730507 = messageFormatterFn((function(  ) {
  return function(d) { return "Background"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1243611932 = messageFormatterFn((function(  ) {
  return function(d) { return "Opacity"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1451639454 = messageFormatterFn((function(  ) {
  return function(d) { return "%"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b605462971 = messageFormatterFn((function(  ) {
  return function(d) { return "characters"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b393144260 = messageFormatterFn((function(  ) {
  return function(d) { return "Background Opacity Slider"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1484069256 = messageFormatterFn((function(  ) {
  return function(d) { return "Button Style"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a788314400 = messageFormatterFn((function(  ) {
  return function(d) { return "Image"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1531763702 = messageFormatterFn((function(  ) {
  return function(d) { return "Text"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b487156737 = messageFormatterFn((function(  ) {
  return function(d) { return "Title"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b53947005 = messageFormatterFn((function(  ) {
  return function(d) { return "Enter optional title"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a7650843 = messageFormatterFn((function(  ) {
  return function(d) { return "Button Appearance"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b419230001 = messageFormatterFn((function(  ) {
  return function(d) { return "Item Shown"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1882025923 = messageFormatterFn((function(  ) {
  return function(d) { return "Item Hidden"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1860084446 = messageFormatterFn((function(  ) {
  return function(d) { return "Font"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1250645460 = messageFormatterFn((function(  ) {
  return function(d) { return "Edit Button"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1144723037 = messageFormatterFn((function(  ) {
  return function(d) { return "Show template project contributors!"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a535706594 = messageFormatterFn((function(  ) {
  return function(d) { return "Hide template project contributors!"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a835228329 = messageFormatterFn((function(  ) {
  return function(d) { return "Warning"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1066741076 = messageFormatterFn((function(  ) {
  return function(d) { return "Add"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1630707529 = messageFormatterFn((function(  ) {
  return function(d) { return "Cancel"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a465686873 = messageFormatterFn((function(  ) {
  return function(d) { return "Learn more"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2131227024 = messageFormatterFn((function(  ) {
  return function(d) { return "Don't show this again"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a120113005 = messageFormatterFn((function(  ) {
  return function(d) { return "Details"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1540969910 = messageFormatterFn((function(  ) {
  return function(d) { return "Field to be added"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1553034028 = messageFormatterFn((function(  ) {
  return function(d) { return "Fields with Relatedness Issues"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2018643845 = messageFormatterFn((function(  ) {
  return function(d) { return d.text + "…"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1102002006 = messageFormatterFn((function(  ) {
  return function(d) { return "Reset"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b788295243 = messageFormatterFn((function(  ) {
  return function(d) { return "Apply"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a29554857 = messageFormatterFn((function(  ) {
  return function(d) { return "Cancel"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a96225643 = messageFormatterFn((function(  ) {
  return function(d) { return "OK"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2113481070 = messageFormatterFn((function(  ) {
  return function(d) { return "Enter a valid data value."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a972441498 = messageFormatterFn((function(  ) {
  return function(d) { return "Delete"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2057769810 = messageFormatterFn((function(  ) {
  return function(d) { return "All values in context"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a781530958 = messageFormatterFn((function(  ) {
  return function(d) { return "All values in data set"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b378578930 = messageFormatterFn((function(  ) {
  return function(d) { return "Only relevant values"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a442196372 = messageFormatterFn((function(  ) {
  return function(d) { return "All values in hierarchy"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b356380911 = messageFormatterFn((function(  ) {
  return function(d) { return "All values in hierarchy and context"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a442052659 = messageFormatterFn((function(  ) {
  return function(d) { return "Select from list"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2024678533 = messageFormatterFn((function(  ) {
  return function(d) { return "Custom value list"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b158286647 = messageFormatterFn((function(  ) {
  return function(d) { return "Use all"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2125848097 = messageFormatterFn((function(  ) {
  return function(d) { return "Exclude selected values"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1240765687 = messageFormatterFn((function(  ) {
  return function(d) { return "Wildcard"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1856746984 = messageFormatterFn((function(  ) {
  return function(d) { return "define wildcard settings"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1457635806 = messageFormatterFn((function(  ) {
  return function(d) { return "Condition"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1856492737 = messageFormatterFn((function(  ) {
  return function(d) { return "define condition settings"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a653223847 = messageFormatterFn((function(  ) {
  return function(d) { return "Top/Bottom"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1142102182 = messageFormatterFn((function(  ) {
  return function(d) { return "define top and bottom settings"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2009547033 = messageFormatterFn((function(  ) {
  return function(d) { return "None"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a632857834 = messageFormatterFn((function(  ) {
  return function(d) { return "By field"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1142865642 = messageFormatterFn((function(  ) {
  return function(d) { return "By formula"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2145273477 = messageFormatterFn((function(  ) {
  return function(d) { return "In a browser, you can't edit the formula. Use Tableau Desktop instead."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a678513020 = messageFormatterFn((function(  ) {
  return function(d) { return "Selection:"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1951020422 = messageFormatterFn((function(  ) {
  return function(d) { return "Wildcard:"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a954257037 = messageFormatterFn((function(  ) {
  return function(d) { return "Condition:"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b585185811 = messageFormatterFn((function(  ) {
  return function(d) { return "Limit:"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1111932856 = messageFormatterFn((function(  ) {
  return function(d) { return "Include all values when empty"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1938109027 = messageFormatterFn((function(  ) {
  return function(d) { return "Filter to latest date value when workbook is opened"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1052229066 = messageFormatterFn((function(  ) {
  return function(d) { return "Loading all values..."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1426973947 = messageFormatterFn((function(  ) {
  return function(d) { return "Add custom values"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1713097386 = messageFormatterFn((function(  ) {
  return function(d) { return "Loading..."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1316758799 = messageFormatterFn((function(  ) {
  return function(d) { return "(All)"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a458527806 = messageFormatterFn((function(  ) {
  return function(d) { return "(All in Search)"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b166813294 = messageFormatterFn((function(  ) {
  return function(d) { return "Clear List"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1379319164 = messageFormatterFn((function(  ) {
  return function(d) { return "No matches"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a726629807 = messageFormatterFn((function(  ) {
  return function(d) { return "General"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1785652654 = messageFormatterFn((function(  ) {
  return function(d) { return "define general settings"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1953522925 = messageFormatterFn((function(  ) {
  return function(d) { return "Summary"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1081069490 = messageFormatterFn((function(  ) {
  return function(d) { return "Search"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1800928374 = messageFormatterFn((function(  ) {
  return function(d) { return "Search or add text"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2070440479 = messageFormatterFn((function(  ) {
  return function(d) { return "Add text"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b812691927 = messageFormatterFn((function(  ) {
  return function(d) { return "Search"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b453084614 = messageFormatterFn((function(  ) {
  return function(d) { return "Add"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1306209060 = messageFormatterFn((function(  ) {
  return function(d) { return "Top"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1985332198 = messageFormatterFn((function(  ) {
  return function(d) { return "Bottom"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1053490804 = messageFormatterFn((function(  ) {
  return function(d) { return "Create a New Parameter..."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1051768591 = messageFormatterFn((function(  ) {
  return function(d) { return "The filter limit must be greater than zero."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a873437358 = messageFormatterFn((function(  ) {
  return function(d) { return "This value isn’t formatted correctly."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a890547726 = messageFormatterFn((function(  ) {
  return function(d) { return "Contains"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b557594615 = messageFormatterFn((function(  ) {
  return function(d) { return "Does not contain"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1258466906 = messageFormatterFn((function(  ) {
  return function(d) { return "Starts with"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b944867331 = messageFormatterFn((function(  ) {
  return function(d) { return "Does not start with"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a883113997 = messageFormatterFn((function(  ) {
  return function(d) { return "Ends with"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1566983222 = messageFormatterFn((function(  ) {
  return function(d) { return "Does not end with"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1054218042 = messageFormatterFn((function(  ) {
  return function(d) { return "Exactly matches"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2117948614 = messageFormatterFn((function(  ) {
  return function(d) { return "Does not match"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1226620488 = messageFormatterFn((function(  ) {
  return function(d) { return "Empty formula"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a962755350 = messageFormatterFn((function(  ) {
  return function(d) { return "Count"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b856327103 = messageFormatterFn((function(  ) {
  return function(d) { return "Field"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b748060887 = messageFormatterFn((function(  ) {
  return function(d) { return "Aggregation"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2144863060 = messageFormatterFn((function(  ) {
  return function(d) { return "Percentile"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2025480531 = messageFormatterFn((function(  ) {
  return function(d) { return "Formula"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b788966280 = messageFormatterFn((function(  ) {
  return function(d) { return "Comparison"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a236530424 = messageFormatterFn((function(  ) {
  return function(d) { return "Value"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a534996571 = messageFormatterFn((function(  ) {
  return function(d) { return "Match value"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1907953957 = messageFormatterFn((function(  ) {
  return function(d) { return "Rank"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a734137518 = messageFormatterFn((function(  ) {
  return function(d) { return "Equals"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b87116245 = messageFormatterFn((function(  ) {
  return function(d) { return "Not equal to"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1120279543 = messageFormatterFn((function(  ) {
  return function(d) { return "Less than"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b232461575 = messageFormatterFn((function(  ) {
  return function(d) { return "Less than or equal to"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1991350590 = messageFormatterFn((function(  ) {
  return function(d) { return "Greater than"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1360357472 = messageFormatterFn((function(  ) {
  return function(d) { return "Greater than or equal to"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1566958979 = messageFormatterFn((function(  ) {
  return function(d) { return "Range of values"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1258607445 = messageFormatterFn((function(  ) {
  return function(d) { return "Minimum"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a486128871 = messageFormatterFn((function(  ) {
  return function(d) { return "Maximum"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2072183091 = messageFormatterFn((function(  ) {
  return function(d) { return "List"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1257415343 = messageFormatterFn((function(  ) {
  return function(d) { return "Values"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2008803250 = messageFormatterFn((function(  ) {
  return function(d) { return "Other"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1785102427 = messageFormatterFn((function(  ) {
  return function(d) { return "Certification icon"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1244229286 = messageFormatterFn((function(  ) {
  return function(d) { return "Certified by"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1344488684 = messageFormatterFn((function(  ) {
  return function(d) { return "Certified Datasource"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b156444512 = messageFormatterFn((function(  ) {
  return function(d) { return "Certified Datasource (Live)"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1758048177 = messageFormatterFn((function(  ) {
  return function(d) { return "Data connection icon"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a882984915 = messageFormatterFn((function(  ) {
  return function(d) { return "External server connection icon"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b236621279 = messageFormatterFn((function(  ) {
  return function(d) { return "File icon"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2002208184 = messageFormatterFn((function(  ) {
  return function(d) { return "Live connection"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b217021229 = messageFormatterFn((function(  ) {
  return function(d) { return "Last extract:"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a888757650 = messageFormatterFn((function(  ) {
  return function(d) { return "No data connections"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1764201384 = messageFormatterFn((function(  ) {
  return function(d) { return "Details"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1801004092 = messageFormatterFn((function(  ) {
  return function(d) { return "Tableau Server connection icon"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a470074019 = messageFormatterFn((function(  ) {
  return function(d) { return "Certified Tableau Server connection icon"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1055494454 = messageFormatterFn((function(  ) {
  return function(d) { return "This workbook connects to"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a669842600 = messageFormatterFn((function(  ) {
  return function(d) { return "Add an Extension"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1152334056 = messageFormatterFn((function(  ) {
  return function(d) { return "We were unable to reach the Tableau Exchange. Please confirm you are connected to the internet."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a331736982 = messageFormatterFn((function(  ) {
  return function(d) { return "You can still use any extensions that you have downloaded."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b182619056 = messageFormatterFn((function(  ) {
  return function(d) { return "The Tableau Exchange is not available in this browser."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a25492626 = messageFormatterFn((function(  ) {
  return function(d) { return "File reading aborted"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b285286090 = messageFormatterFn((function(  ) {
  return function(d) { return "Selected file is too large. Select a file less than " + d.fileSize + "MB"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1124602512 = messageFormatterFn((function(  ) {
  return function(d) { return "Invalid Extension Manifest selected. Only .trex files are supported."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1482451187 = messageFormatterFn((function(  ) {
  return function(d) { return "My Extensions"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2012804157 = messageFormatterFn((function(  ) {
  return function(d) { return "Wrong number of files selected"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1985695964 = messageFormatterFn((function(  ) {
  return function(d) { return "Show template project contributors!"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1376679521 = messageFormatterFn((function(  ) {
  return function(d) { return "Hide template project contributors!"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1336442465 = messageFormatterFn((function(  ) {
  return function(d) { return "Cancel"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1719281273 = messageFormatterFn((function(  ) {
  return function(d) { return "Connect"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1899886773 = messageFormatterFn((function(  ) {
  return function(d) { return "Use Endpoint"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1151567985 = messageFormatterFn((function(  ) {
  return function(d) { return "List Endpoints"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2025330659 = messageFormatterFn((function(  ) {
  return function(d) { return "File"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1753726805 = messageFormatterFn((function(  ) {
  return function(d) { return "Folder"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1897219346 = messageFormatterFn((function(  ) {
  return function(d) { return "Site"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1136161817 = messageFormatterFn((function(  ) {
  return function(d) { return "Document Library"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1315449612 = messageFormatterFn((function(  ) {
  return function(d) { return "Shared Drive"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2088159878 = messageFormatterFn((function(  ) {
  return function(d) { return "Home"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2126432261 = messageFormatterFn((function(  ) {
  return function(d) { return "Kind"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b290255707 = messageFormatterFn((function(  ) {
  return function(d) { return "Last Modified On"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1897040600 = messageFormatterFn((function(  ) {
  return function(d) { return "Size"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2047961806 = messageFormatterFn((function(  ) {
  return function(d) { return "Name"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1840312086 = messageFormatterFn((function(  ) {
  return function(d) { return "No item matching the searched text were found"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1946705393 = messageFormatterFn((function(  ) {
  return function(d) { return "Search"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1413970616 = messageFormatterFn((function(  ) {
  return function(d) { return "Search site names"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a347448864 = messageFormatterFn((function(  ) {
  return function(d) { return "Search Results"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1607928550 = messageFormatterFn((function(  ) {
  return function(d) { return "signed in as " + d.user + " "; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b939486176 = messageFormatterFn((function(  ) {
  return function(d) { return "sign out"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1023705213 = messageFormatterFn((function(  ) {
  return function(d) { return "The URL you entered is invalid or points to a file you don't have permission to view"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1897898156 = messageFormatterFn((function(  ) {
  return function(d) { return "Browse"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a944307487 = messageFormatterFn((function(  ) {
  return function(d) { return "File URL"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1529063141 = messageFormatterFn((function(  ) {
  return function(d) { return "Endpoint URL"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1390976821 = messageFormatterFn((function(  ) {
  return function(d) { return "File URL"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b611060110 = messageFormatterFn((function(  ) {
  return function(d) { return "File extension:"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1786352382 = messageFormatterFn((function(  ) {
  return function(d) { return "File URL must start with " + d.urlBase; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a829138476 = messageFormatterFn((function(  ) {
  return function(d) { return "Endpoint must be in the format of https://{accountName}.dfs.core.windows.net/"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1867904693 = messageFormatterFn((function(  ) {
  return function(d) { return "<p><strong>Can't authorize access</strong><br />Tableau can’t authorize access to the endpoint.</p><p>What to do now:<ul><li>Make sure you have permission to access, then try again. <a target='_blank' href='https://help.tableau.com/current/pro/desktop/en-us/examples_azure_data_lake_gen2.htm'>Learn more</a>.</li><li>Close this connector and try again.</li><ul></p>"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1488100742 = messageFormatterFn((function(  ) {
  return function(d) { return "OneDrive and SharePoint Online"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b779851893 = messageFormatterFn((function(  ) {
  return function(d) { return "OneDrive (personal files)"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1683821456 = messageFormatterFn((function(  ) {
  return function(d) { return "OneDrive (shared with you)"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1894764778 = messageFormatterFn((function(  ) {
  return function(d) { return "SharePoint Sites"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1632097347 = messageFormatterFn((function(  ) {
  return function(d) { return "SharePoint Site URL should be in format of https://{host}.sharepoint.com or https://{host}.sharepoint.com/sites/{path/to/site}."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1755908791 = messageFormatterFn((function(  ) {
  return function(d) { return "Site URL"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b8769962 = messageFormatterFn((function(  ) {
  return function(d) { return "Browse This Site"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b643499107 = messageFormatterFn((function(  ) {
  return function(d) { return "List Sites"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b843585452 = messageFormatterFn((function(  ) {
  return function(d) { return "My Drive"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b526311869 = messageFormatterFn((function(  ) {
  return function(d) { return "Shared with me"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b183634258 = messageFormatterFn((function(  ) {
  return function(d) { return "Shared drives"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a402237161 = messageFormatterFn((function(  ) {
  return function(d) { return "Show template project contributors!"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b206779282 = messageFormatterFn((function(  ) {
  return function(d) { return "Hide template project contributors!"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1111282783 = messageFormatterFn((function(  ) {
  return function(d) { return "@mention to notify someone"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a856642330 = messageFormatterFn((function(  ) {
  return function(d) { return "Add a comment"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a586234441 = messageFormatterFn((function(  ) {
  return function(d) { return "Add a snapshot of the view to your comment"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b417380523 = messageFormatterFn((function(  ) {
  return function(d) { return "Couldn't create a snapshot of the view"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a955404030 = messageFormatterFn((function(  ) {
  return function(d) { return "You don't have permission to add a snapshot of this view."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1465273848 = messageFormatterFn((function(  ) {
  return function(d) { return "Applying the snapshot's filters and selection to the view..."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1885602008 = messageFormatterFn((function(  ) {
  return function(d) { return "Couldn't apply snapshot from " + d.user + " to the view"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1001526983 = messageFormatterFn((function(  ) {
  return function(d) { return "Press Enter or Space to apply the snapshot filters to the view."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a937118726 = messageFormatterFn((function(  ) {
  return function(d) { return "Something went wrong"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a544430526 = messageFormatterFn((function(  ) {
  return function(d) { return "Close panel"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a169003142 = messageFormatterFn((function(  ) {
  return function(d) { return "Comments"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a253711618 = messageFormatterFn((function(  ) {
  return function(d) { return "You no longer have add comment permission."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b127737520 = messageFormatterFn((function( plural, pluralFuncs ) {
  return function(d) { return plural(d.itemCount, 0, pluralFuncs.en, { one: function() { return "User";}, other: function() { return d.formattedItemCount + " users";} }) + " will not be notified about " + d.comment; }
})(messageFormat.plural, {"en": Globalize("en").pluralGenerator()}), Globalize("en").pluralGenerator({}));
Globalize.a1471531500 = messageFormatterFn((function(  ) {
  return function(d) { return "You no longer have permission to view the comments."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1406998927 = messageFormatterFn((function(  ) {
  return function(d) { return "Confirm delete of comment by " + d.user + " on " + d.date; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1524561682 = messageFormatterFn((function(  ) {
  return function(d) { return "Check your internet connection and retry."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b904005292 = messageFormatterFn((function(  ) {
  return function(d) { return "Check your internet connection and refresh to try again"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2069839777 = messageFormatterFn((function(  ) {
  return function(d) { return "Contact your administrator."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a18627952 = messageFormatterFn((function(  ) {
  return function(d) { return "Cancel"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b103099487 = messageFormatterFn((function(  ) {
  return function(d) { return "Delete"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1995720738 = messageFormatterFn((function(  ) {
  return function(d) { return "This message has already been deleted."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1713758616 = messageFormatterFn((function(  ) {
  return function(d) { return "You do not have permission to delete this comment."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a240856357 = messageFormatterFn((function(  ) {
  return function(d) { return "Press Delete or Backspace to delete this comment."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1423669119 = messageFormatterFn((function(  ) {
  return function(d) { return "Delete"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a490842535 = messageFormatterFn((function(  ) {
  return function(d) { return "Failed to download image"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1117524006 = messageFormatterFn((function(  ) {
  return function(d) { return "An unknown error has occurred"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1513402832 = messageFormatterFn((function(  ) {
  return function(d) { return "Failed to get comments"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a333220967 = messageFormatterFn((function(  ) {
  return function(d) { return "Failed to get image"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1591409219 = messageFormatterFn((function(  ) {
  return function(d) { return "Learn more"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1104549106 = messageFormatterFn((function(  ) {
  return function(d) { return "Unable to load comments"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2089505580 = messageFormatterFn((function(  ) {
  return function(d) { return "Loading comments..."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1250654285 = messageFormatterFn((function(  ) {
  return function(d) { return "Message is too long"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1015530127 = messageFormatterFn((function(  ) {
  return function(d) { return "No comments on this view."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b229925326 = messageFormatterFn((function(  ) {
  return function(d) { return "@mention someone in your comment to notify them"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1889096782 = messageFormatterFn((function(  ) {
  return function(d) { return "Post"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b163283605 = messageFormatterFn((function(  ) {
  return function(d) { return "Remove snapshot"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1237763732 = messageFormatterFn((function(  ) {
  return function(d) { return "Retry"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1392247716 = messageFormatterFn((function(  ) {
  return function(d) { return "Couldn't send."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1430985120 = messageFormatterFn((function(  ) {
  return function(d) { return "Snapshot added."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1504116337 = messageFormatterFn((function(  ) {
  return function(d) { return "A snapshot of the view is attached to this comment."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2118798872 = messageFormatterFn((function(  ) {
  return function(d) { return "Snapshot preview"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b3792768 = messageFormatterFn((function(  ) {
  return function(d) { return "Snapshot removed."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1523397989 = messageFormatterFn((function(  ) {
  return function(d) { return "Tap to retry"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b274322310 = messageFormatterFn((function(  ) {
  return function(d) { return "Unable to load comments"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1861665231 = messageFormatterFn((function(  ) {
  return function(d) { return d.user + " doesn't have permissions to see this view"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a221714838 = messageFormatterFn((function( plural, pluralFuncs ) {
  return function(d) { return plural(d.itemCount, 0, pluralFuncs.en, { one: function() { return d.formattedItemCount + " user";}, other: function() { return d.formattedItemCount + " users";} }) + " won't be notified"; }
})(messageFormat.plural, {"en": Globalize("en").pluralGenerator()}), Globalize("en").pluralGenerator({}));
Globalize.b1318343138 = messageFormatterFn((function(  ) {
  return function(d) { return d.user + " lacks an email address in their Tableau account"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b481362245 = messageFormatterFn((function(  ) {
  return function(d) { return d.user + " doesn’t have permissions to see comments"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1797312956 = messageFormatterFn((function(  ) {
  return function(d) { return "User will not be notified"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a631426777 = messageFormatterFn((function(  ) {
  return function(d) { return "User Removed"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2079754917 = messageFormatterFn((function(  ) {
  return function(d) { return "View"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1406696097 = messageFormatterFn((function(  ) {
  return function(d) { return "All Tables"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a685160792 = messageFormatterFn((function(  ) {
  return function(d) { return "Columns"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2042918107 = messageFormatterFn((function(  ) {
  return function(d) { return "Connect"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b366009847 = messageFormatterFn((function(  ) {
  return function(d) { return "Data sources"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a883144759 = messageFormatterFn((function(  ) {
  return function(d) { return "Database / Schema"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1329131196 = messageFormatterFn((function(  ) {
  return function(d) { return "Default"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1356903710 = messageFormatterFn((function(  ) {
  return function(d) { return "Expand / Collapse"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1949906422 = messageFormatterFn((function(  ) {
  return function(d) { return "Filters"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b879013882 = messageFormatterFn((function(  ) {
  return function(d) { return "Name"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1269410266 = messageFormatterFn((function(  ) {
  return function(d) { return "Salesforce"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a425785315 = messageFormatterFn((function(  ) {
  return function(d) { return "Search"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a690468678 = messageFormatterFn((function(  ) {
  return function(d) { return "Search for data"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a376703156 = messageFormatterFn((function(  ) {
  return function(d) { return "Workbooks"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1682182902 = messageFormatterFn((function(  ) {
  return function(d) { return "Remember Password"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b33920924 = messageFormatterFn((function(  ) {
  return function(d) { return "Authentication:"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a125071357 = messageFormatterFn((function(  ) {
  return function(d) { return "Microsoft Azure HDInsight Service"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1323979673 = messageFormatterFn((function(  ) {
  return function(d) { return "Integrated Authentication"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2061365663 = messageFormatterFn((function(  ) {
  return function(d) { return "Kerberos"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b303434483 = messageFormatterFn((function(  ) {
  return function(d) { return "LDAP"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b205104002 = messageFormatterFn((function(  ) {
  return function(d) { return "No Authentication"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1809250311 = messageFormatterFn((function(  ) {
  return function(d) { return "Sign in using OAuth"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1757495000 = messageFormatterFn((function(  ) {
  return function(d) { return "SAML IdP"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b466720944 = messageFormatterFn((function(  ) {
  return function(d) { return "Teradata Database"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a311133740 = messageFormatterFn((function(  ) {
  return function(d) { return "Username and Password"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b813803351 = messageFormatterFn((function(  ) {
  return function(d) { return "Active Directory Password"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1109256476 = messageFormatterFn((function(  ) {
  return function(d) { return "Username"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a218675035 = messageFormatterFn((function(  ) {
  return function(d) { return "Windows Authentication"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a54050209 = messageFormatterFn((function(  ) {
  return function(d) { return "Password"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2080762645 = messageFormatterFn((function(  ) {
  return function(d) { return "Direct"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1366923348 = messageFormatterFn((function(  ) {
  return function(d) { return "HiveServer2"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1399224630 = messageFormatterFn((function(  ) {
  return function(d) { return "Impala"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a489929650 = messageFormatterFn((function(  ) {
  return function(d) { return "SharkServer (Shark 0.8.1 and earlier)"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1992036394 = messageFormatterFn((function(  ) {
  return function(d) { return "SharkServer2 (Shark 0.9.*)"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1522591402 = messageFormatterFn((function(  ) {
  return function(d) { return "SparkThriftServer (Spark 1.1 and later)"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1091946038 = messageFormatterFn((function(  ) {
  return function(d) { return "ZooKeeper"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2034559786 = messageFormatterFn((function(  ) {
  return function(d) { return "SingleNode"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a26827269 = messageFormatterFn((function(  ) {
  return function(d) { return "MultiNode"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a626249337 = messageFormatterFn((function(  ) {
  return function(d) { return "Add SQL statements to be run at connect time."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a9829008 = messageFormatterFn((function(  ) {
  return function(d) { return "Learn more"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a339708011 = messageFormatterFn((function(  ) {
  return function(d) { return "Initial SQL"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b956419018 = messageFormatterFn((function(  ) {
  return function(d) { return "Insert"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1320678518 = messageFormatterFn((function(  ) {
  return function(d) { return "Show Initial SQL"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1707293669 = messageFormatterFn((function(  ) {
  return function(d) { return "Hide Initial SQL"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a634131777 = messageFormatterFn((function(  ) {
  return function(d) { return "Binary"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1510905128 = messageFormatterFn((function(  ) {
  return function(d) { return "HTTP"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1708409848 = messageFormatterFn((function(  ) {
  return function(d) { return "No Transport Type"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1808245255 = messageFormatterFn((function(  ) {
  return function(d) { return "SASL"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a313927897 = messageFormatterFn((function(  ) {
  return function(d) { return "Select a DSN (data source name) for the database you want to connect to. To publish and run your flow on Tableau Server, the server must be configured with a matching DSN."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1083647901 = messageFormatterFn((function(  ) {
  return function(d) { return "DSN:"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1091240746 = messageFormatterFn((function(  ) {
  return function(d) { return "Select DSN"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a824791310 = messageFormatterFn((function(  ) {
  return function(d) { return "Copy"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1358687201 = messageFormatterFn((function(  ) {
  return function(d) { return "Cut"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1946751056 = messageFormatterFn((function(  ) {
  return function(d) { return "Paste"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b637441164 = messageFormatterFn((function(  ) {
  return function(d) { return "Read uncommitted data"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1930901595 = messageFormatterFn((function(  ) {
  return function(d) { return "Require Encryption"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b925271808 = messageFormatterFn((function(  ) {
  return function(d) { return "Require SSL (recommended)"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1181860937 = messageFormatterFn((function(  ) {
  return function(d) { return "Keep your data safe by encrypting the data connection"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1361585235 = messageFormatterFn((function(  ) {
  return function(d) { return "SSL certificate file"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b87476619 = messageFormatterFn((function(  ) {
  return function(d) { return "Detailed Error Message"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b935903431 = messageFormatterFn((function(  ) {
  return function(d) { return "Optional"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1201480561 = messageFormatterFn((function(  ) {
  return function(d) { return "Sign In"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b544606338 = messageFormatterFn((function(  ) {
  return function(d) { return "Signing In…"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b384954460 = messageFormatterFn((function(  ) {
  return function(d) { return "Type:"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b956035642 = messageFormatterFn((function(  ) {
  return function(d) { return "Username:"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1283605601 = messageFormatterFn((function(  ) {
  return function(d) { return "Password:"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1425838799 = messageFormatterFn((function(  ) {
  return function(d) { return "Access Key ID:"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1704455350 = messageFormatterFn((function(  ) {
  return function(d) { return "Secret Access Key:"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2053829712 = messageFormatterFn((function(  ) {
  return function(d) { return "SAML IdP(Okta):"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1079906130 = messageFormatterFn((function(  ) {
  return function(d) { return d.prompt + ":"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b709024273 = messageFormatterFn((function(  ) {
  return function(d) { return "HTTP:"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b284692636 = messageFormatterFn((function(  ) {
  return function(d) { return "Realm:"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1306109378 = messageFormatterFn((function(  ) {
  return function(d) { return "Service Name:"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1564067161 = messageFormatterFn((function(  ) {
  return function(d) { return "Host FQDN:"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a247014005 = messageFormatterFn((function(  ) {
  return function(d) { return "Transport:"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a535954302 = messageFormatterFn((function(  ) {
  return function(d) { return "OAuth Service URL:"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2117222343 = messageFormatterFn((function(  ) {
  return function(d) { return "Connect"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b968910689 = messageFormatterFn((function(  ) {
  return function(d) { return "To a File"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1294093658 = messageFormatterFn((function(  ) {
  return function(d) { return "To a Server"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1342160884 = messageFormatterFn((function(  ) {
  return function(d) { return "URL:"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b267063805 = messageFormatterFn((function(  ) {
  return function(d) { return "Dialect:"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b516367101 = messageFormatterFn((function(  ) {
  return function(d) { return "Properties File:"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a629610943 = messageFormatterFn((function(  ) {
  return function(d) { return "Hide connection details"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1558801372 = messageFormatterFn((function(  ) {
  return function(d) { return "Show connection details"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a897082197 = messageFormatterFn((function(  ) {
  return function(d) { return "Filename:"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1707716302 = messageFormatterFn((function(  ) {
  return function(d) { return "Database password:"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b173858064 = messageFormatterFn((function(  ) {
  return function(d) { return "Workgroup security"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b355575117 = messageFormatterFn((function(  ) {
  return function(d) { return "Workgroup file:"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1783673374 = messageFormatterFn((function(  ) {
  return function(d) { return "Username:"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a455967869 = messageFormatterFn((function(  ) {
  return function(d) { return "Password:"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1052854429 = messageFormatterFn((function(  ) {
  return function(d) { return "Browse"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1422042490 = messageFormatterFn((function(  ) {
  return function(d) { return "Browse…"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1551324537 = messageFormatterFn((function(  ) {
  return function(d) { return "Open"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1897872952 = messageFormatterFn((function(  ) {
  return function(d) { return "No file chosen"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a640388178 = messageFormatterFn((function(  ) {
  return function(d) { return "For support, contact vendor."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b929718168 = messageFormatterFn((function(  ) {
  return function(d) { return "Download and install the drivers,"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1059726964 = messageFormatterFn((function(  ) {
  return function(d) { return " and then connect."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a842460618 = messageFormatterFn((function(  ) {
  return function(d) { return "No driver installed for " + d.connectorName + " connector. Contact your administrator."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1219655132 = messageFormatterFn((function(  ) {
  return function(d) { return "Edit Custom SQL"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1095711959 = messageFormatterFn((function(  ) {
  return function(d) { return "Convert to SQL"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a368823831 = messageFormatterFn((function(  ) {
  return function(d) { return "OK"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b690235051 = messageFormatterFn((function(  ) {
  return function(d) { return "Cancel"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1088246857 = messageFormatterFn((function(  ) {
  return function(d) { return "Enter SQL query here. For example, 'SELECT * FROM table_name'"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1297433604 = messageFormatterFn((function(  ) {
  return function(d) { return "Insert Parameter:"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1668497494 = messageFormatterFn((function(  ) {
  return function(d) { return "Create a New Parameter..."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1144743934 = messageFormatterFn((function(  ) {
  return function(d) { return "Preview Results..."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a190745625 = messageFormatterFn((function(  ) {
  return function(d) { return "Original"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b503745984 = messageFormatterFn((function(  ) {
  return function(d) { return "Author:"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1643141149 = messageFormatterFn((function(  ) {
  return function(d) { return "Name this view"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b458854702 = messageFormatterFn((function(  ) {
  return function(d) { return "Save"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2056860341 = messageFormatterFn((function(  ) {
  return function(d) { return "Please enter a name for the Custom View in the text field."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1898957137 = messageFormatterFn((function(  ) {
  return function(d) { return "Make it my default"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1126302854 = messageFormatterFn((function(  ) {
  return function(d) { return "Make visible to others"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a689237299 = messageFormatterFn((function(  ) {
  return function(d) { return "Anyone with access to the view can see this custom view"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1936029142 = messageFormatterFn((function(  ) {
  return function(d) { return "Save Custom View"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a357933758 = messageFormatterFn((function(  ) {
  return function(d) { return "Save your own custom view with all filters applied, to eliminate the need to filter every time you open the dashboard. "; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1392985580 = messageFormatterFn((function(  ) {
  return function(d) { return "Learn more"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1556463350 = messageFormatterFn((function(  ) {
  return function(d) { return "Select the view name to change the view or use the icons to modify the view."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b901055979 = messageFormatterFn((function(  ) {
  return function(d) { return "My Views"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1972842817 = messageFormatterFn((function(  ) {
  return function(d) { return "Other Views"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a690617898 = messageFormatterFn((function(  ) {
  return function(d) { return " (default)"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a825625707 = messageFormatterFn((function(  ) {
  return function(d) { return "Nothing saved yet"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b557432134 = messageFormatterFn((function(  ) {
  return function(d) { return "Save a custom view for quick access to your preferred filters."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1990151377 = messageFormatterFn((function(  ) {
  return function(d) { return "Manage Views"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a723552517 = messageFormatterFn((function(  ) {
  return function(d) { return "Back"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a45172943 = messageFormatterFn((function(  ) {
  return function(d) { return "Manage Views"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1848417665 = messageFormatterFn((function(  ) {
  return function(d) { return "Select this view as your current view"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b118725386 = messageFormatterFn((function(  ) {
  return function(d) { return "This view is your current view"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b814606043 = messageFormatterFn((function(  ) {
  return function(d) { return "Set this view as your default"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b179230064 = messageFormatterFn((function(  ) {
  return function(d) { return "This is your current default"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b69032058 = messageFormatterFn((function(  ) {
  return function(d) { return "Rename this view"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b365949794 = messageFormatterFn((function(  ) {
  return function(d) { return "Rename this view"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1805206622 = messageFormatterFn((function(  ) {
  return function(d) { return "This view is private"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b98992704 = messageFormatterFn((function(  ) {
  return function(d) { return "This view is public"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b149089927 = messageFormatterFn((function(  ) {
  return function(d) { return "Delete this view"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b454275715 = messageFormatterFn((function(  ) {
  return function(d) { return "Cancel"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a488610926 = messageFormatterFn((function(  ) {
  return function(d) { return "Delete"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1263412592 = messageFormatterFn((function(  ) {
  return function(d) { return "Are you sure?"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b966084267 = messageFormatterFn((function(  ) {
  return function(d) { return "Enter a name"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b565965880 = messageFormatterFn((function(  ) {
  return function(d) { return "Max character limit is 255"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1796101249 = messageFormatterFn((function(  ) {
  return function(d) { return d.VIEW_NAME + ": " + d.PROPERTY + "."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1969331971 = messageFormatterFn((function(  ) {
  return function(d) { return d.VIEW_NAME + ": " + d.FIRST_PROPERTY + ". " + d.SECOND_PROPERTY + "."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1671531062 = messageFormatterFn((function(  ) {
  return function(d) { return "Actions let you create interactive relationships between data, dashboard objects, other worksheets, and the web."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1490018517 = messageFormatterFn((function(  ) {
  return function(d) { return "Show actions for"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b29296405 = messageFormatterFn((function(  ) {
  return function(d) { return "This sheet"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2012977203 = messageFormatterFn((function(  ) {
  return function(d) { return "This workbook"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1449181573 = messageFormatterFn((function(  ) {
  return function(d) { return "Add Action"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1196822509 = messageFormatterFn((function(  ) {
  return function(d) { return "Filter..."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1401089942 = messageFormatterFn((function(  ) {
  return function(d) { return "Add Filter Action"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1992861022 = messageFormatterFn((function(  ) {
  return function(d) { return "True"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1607504309 = messageFormatterFn((function(  ) {
  return function(d) { return "False"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1226283712 = messageFormatterFn((function(  ) {
  return function(d) { return "Create a New Parameter..."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2122629041 = messageFormatterFn((function(  ) {
  return function(d) { return "Edit Filter Action"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1941730846 = messageFormatterFn((function(  ) {
  return function(d) { return "Add Highlight Action"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1413218279 = messageFormatterFn((function(  ) {
  return function(d) { return "Edit Highlight Action"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a567688477 = messageFormatterFn((function(  ) {
  return function(d) { return "Add URL Action"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a315415266 = messageFormatterFn((function(  ) {
  return function(d) { return "Edit URL Action"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1706075223 = messageFormatterFn((function(  ) {
  return function(d) { return "Add Parameter Action"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a766057052 = messageFormatterFn((function(  ) {
  return function(d) { return "Edit Parameter Action"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a493894800 = messageFormatterFn((function(  ) {
  return function(d) { return "Add Set Action"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a241621589 = messageFormatterFn((function(  ) {
  return function(d) { return "Edit Set Action"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b381567744 = messageFormatterFn((function(  ) {
  return function(d) { return "Edit"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1955895494 = messageFormatterFn((function(  ) {
  return function(d) { return "Remove"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b377686775 = messageFormatterFn((function(  ) {
  return function(d) { return "Go to Sheet..."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b409349644 = messageFormatterFn((function(  ) {
  return function(d) { return "Add Go to Sheet Action"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a451097121 = messageFormatterFn((function(  ) {
  return function(d) { return "Highlight..."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1352792676 = messageFormatterFn((function(  ) {
  return function(d) { return "Go to URL..."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b661622855 = messageFormatterFn((function(  ) {
  return function(d) { return "Edit Go to Sheet Action"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b466580964 = messageFormatterFn((function(  ) {
  return function(d) { return "Change Parameter..."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b298265963 = messageFormatterFn((function(  ) {
  return function(d) { return "Change Set Values..."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a898807092 = messageFormatterFn((function(  ) {
  return function(d) { return "Selected sheets"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2002274096 = messageFormatterFn((function(  ) {
  return function(d) { return "Cancel"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1266171630 = messageFormatterFn((function(  ) {
  return function(d) { return "OK"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b328315690 = messageFormatterFn((function(  ) {
  return function(d) { return "Filter Action"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2073580948 = messageFormatterFn((function(  ) {
  return function(d) { return "Highlight Action"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1384731175 = messageFormatterFn((function(  ) {
  return function(d) { return "Go To Url Action"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1516020439 = messageFormatterFn((function(  ) {
  return function(d) { return "Go To Sheet Action"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1070786593 = messageFormatterFn((function(  ) {
  return function(d) { return "Change Parameter Action"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1513892595 = messageFormatterFn((function(  ) {
  return function(d) { return "Change Set Values Action"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a126964085 = messageFormatterFn((function(  ) {
  return function(d) { return "Name"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1901411760 = messageFormatterFn((function(  ) {
  return function(d) { return "Run On"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2068327239 = messageFormatterFn((function(  ) {
  return function(d) { return "Fields"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a677843237 = messageFormatterFn((function(  ) {
  return function(d) { return "Source"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1913449321 = messageFormatterFn((function(  ) {
  return function(d) { return "In a browser, you can't edit this action; use Tableau Desktop instead."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b466493499 = messageFormatterFn((function(  ) {
  return function(d) { return "Name"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1360441177 = messageFormatterFn((function(  ) {
  return function(d) { return "An empty name is not allowed."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1516295021 = messageFormatterFn((function(  ) {
  return function(d) { return "Select"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2104388115 = messageFormatterFn((function(  ) {
  return function(d) { return "Hover"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2113498832 = messageFormatterFn((function(  ) {
  return function(d) { return "Menu"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b772106921 = messageFormatterFn((function(  ) {
  return function(d) { return "Select"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a173346239 = messageFormatterFn((function(  ) {
  return function(d) { return "Single-select only"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1040521257 = messageFormatterFn((function(  ) {
  return function(d) { return "Hover"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1146806022 = messageFormatterFn((function(  ) {
  return function(d) { return "Menu"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1788789549 = messageFormatterFn((function(  ) {
  return function(d) { return "Date"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b490068896 = messageFormatterFn((function(  ) {
  return function(d) { return "Datetime"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b422556882 = messageFormatterFn((function(  ) {
  return function(d) { return "Number"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1006751428 = messageFormatterFn((function(  ) {
  return function(d) { return "Run action on"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b685759861 = messageFormatterFn((function(  ) {
  return function(d) { return "Clearing the selection will"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b692697860 = messageFormatterFn((function(  ) {
  return function(d) { return "Keep filtered values"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1727713646 = messageFormatterFn((function(  ) {
  return function(d) { return "Keep current value"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a547463231 = messageFormatterFn((function(  ) {
  return function(d) { return "Set value to"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1806630004 = messageFormatterFn((function(  ) {
  return function(d) { return "Show all values"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1491156381 = messageFormatterFn((function(  ) {
  return function(d) { return "Exclude all values"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1974953120 = messageFormatterFn((function(  ) {
  return function(d) { return "Keep set values"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2114628508 = messageFormatterFn((function(  ) {
  return function(d) { return "Add all values to set"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1358414725 = messageFormatterFn((function(  ) {
  return function(d) { return "Remove all values from set"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b470652648 = messageFormatterFn((function(  ) {
  return function(d) { return "Source Sheets"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1276147676 = messageFormatterFn((function(  ) {
  return function(d) { return "Selected source sheets"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2071982478 = messageFormatterFn((function(  ) {
  return function(d) { return "Target Sheets"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2129259482 = messageFormatterFn((function(  ) {
  return function(d) { return "Selected target sheets"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a951577721 = messageFormatterFn((function(  ) {
  return function(d) { return "Select sheets from:"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1878824435 = messageFormatterFn((function(  ) {
  return function(d) { return "Data source"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1556893713 = messageFormatterFn((function(  ) {
  return function(d) { return "Click to add"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b226079052 = messageFormatterFn((function(  ) {
  return function(d) { return "Select all"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a442603951 = messageFormatterFn((function(  ) {
  return function(d) { return "Row " + d.rowIndex; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1574266656 = messageFormatterFn((function(  ) {
  return function(d) { return "Filter"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a15697132 = messageFormatterFn((function(  ) {
  return function(d) { return "All fields"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1304293416 = messageFormatterFn((function(  ) {
  return function(d) { return "Selected fields"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1625955965 = messageFormatterFn((function(  ) {
  return function(d) { return "Source Data Source"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a626911026 = messageFormatterFn((function(  ) {
  return function(d) { return "Source Field"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a916679161 = messageFormatterFn((function(  ) {
  return function(d) { return "Target Data Source"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1458672388 = messageFormatterFn((function(  ) {
  return function(d) { return "Target Field"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a908165834 = messageFormatterFn((function(  ) {
  return function(d) { return "Please specify target field"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b694321443 = messageFormatterFn((function(  ) {
  return function(d) { return "Target"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2997162 = messageFormatterFn((function(  ) {
  return function(d) { return "Target Set"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a736309293 = messageFormatterFn((function(  ) {
  return function(d) { return "Running the action will"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1869206696 = messageFormatterFn((function(  ) {
  return function(d) { return "Assign values to set"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1626452696 = messageFormatterFn((function(  ) {
  return function(d) { return "Add values to set"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b304180444 = messageFormatterFn((function(  ) {
  return function(d) { return "Remove values from set"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b881275041 = messageFormatterFn((function(  ) {
  return function(d) { return "Target Parameter"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1396642448 = messageFormatterFn((function(  ) {
  return function(d) { return "Source Field"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2088333288 = messageFormatterFn((function(  ) {
  return function(d) { return "Aggregation"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1581272462 = messageFormatterFn((function(  ) {
  return function(d) { return "Actions"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a321292330 = messageFormatterFn((function(  ) {
  return function(d) { return "Actions"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1017299300 = messageFormatterFn((function(  ) {
  return function(d) { return "No matches."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1321064114 = messageFormatterFn((function(  ) {
  return function(d) { return "None"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1585840870 = messageFormatterFn((function(  ) {
  return function(d) { return "Insert"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2111767245 = messageFormatterFn((function(  ) {
  return function(d) { return "All Fields"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a184062439 = messageFormatterFn((function(  ) {
  return function(d) { return "Dates and Times"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1066723689 = messageFormatterFn((function(  ) {
  return function(d) { return "Selected Fields"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a521479521 = messageFormatterFn((function(  ) {
  return function(d) { return "Target Highlighting"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b131605005 = messageFormatterFn((function(  ) {
  return function(d) { return "Please select one or more fields to highlight on the target."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b566788613 = messageFormatterFn((function(  ) {
  return function(d) { return "URL"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a655098690 = messageFormatterFn((function(  ) {
  return function(d) { return "Hyperlink input"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1762238433 = messageFormatterFn((function(  ) {
  return function(d) { return "Enter the URL"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b899110242 = messageFormatterFn((function(  ) {
  return function(d) { return "Proceed"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b25811461 = messageFormatterFn((function(  ) {
  return function(d) { return "New Browser Tab"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b926538544 = messageFormatterFn((function(  ) {
  return function(d) { return "Web Page Object"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2057721858 = messageFormatterFn((function(  ) {
  return function(d) { return "New Tab if No Web Page Object Exists"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1813296511 = messageFormatterFn((function(  ) {
  return function(d) { return "URL Target"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a260785757 = messageFormatterFn((function(  ) {
  return function(d) { return "Data Values"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1766276834 = messageFormatterFn((function(  ) {
  return function(d) { return "Learn more"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1936864944 = messageFormatterFn((function(  ) {
  return function(d) { return "Encode data values that URLs do not support"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b149118393 = messageFormatterFn((function(  ) {
  return function(d) { return "Allow multiple values via URL parameters"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b12747174 = messageFormatterFn((function(  ) {
  return function(d) { return "Value Delimiter"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a609697787 = messageFormatterFn((function(  ) {
  return function(d) { return "Delimiter Escape Character"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1584963130 = messageFormatterFn((function(  ) {
  return function(d) { return "Value Delimiter"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a654482425 = messageFormatterFn((function(  ) {
  return function(d) { return "Delimiter Escape Character"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1485438530 = messageFormatterFn((function(  ) {
  return function(d) { return "Please specify a multi-select delimiter."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a929009852 = messageFormatterFn((function(  ) {
  return function(d) { return "Please specify a multi-select escape."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b889011355 = messageFormatterFn((function(  ) {
  return function(d) { return "warning icon"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a214107833 = messageFormatterFn((function(  ) {
  return function(d) { return "error icon"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a416060717 = messageFormatterFn((function(  ) {
  return function(d) { return "dashboard icon"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1694380214 = messageFormatterFn((function(  ) {
  return function(d) { return "data source icon"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a863947027 = messageFormatterFn((function(  ) {
  return function(d) { return "worksheet icon"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a567122974 = messageFormatterFn((function(  ) {
  return function(d) { return "story icon"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2093586034 = messageFormatterFn((function(  ) {
  return function(d) { return "web icon"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1864580537 = messageFormatterFn((function(  ) {
  return function(d) { return "date type icon"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2013157766 = messageFormatterFn((function(  ) {
  return function(d) { return "date time type icon"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a840948329 = messageFormatterFn((function(  ) {
  return function(d) { return "integer type icon"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b65440169 = messageFormatterFn((function(  ) {
  return function(d) { return "real type icon"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b178748796 = messageFormatterFn((function(  ) {
  return function(d) { return "string type icon"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a933486783 = messageFormatterFn((function(  ) {
  return function(d) { return "boolean type icon"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1556808547 = messageFormatterFn((function(  ) {
  return function(d) { return "spatial type icon"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a882157157 = messageFormatterFn((function(  ) {
  return function(d) { return "Above"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1842802659 = messageFormatterFn((function(  ) {
  return function(d) { return "Above or equal to"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b266175882 = messageFormatterFn((function(  ) {
  return function(d) { return "Actions"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1988497984 = messageFormatterFn((function(  ) {
  return function(d) { return "Add Me"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b957523831 = messageFormatterFn((function(  ) {
  return function(d) { return "Alert Invalid"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a799666605 = messageFormatterFn((function(  ) {
  return function(d) { return "Alert Suspended"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b900482457 = messageFormatterFn((function(  ) {
  return function(d) { return "This alert is visible only to recipients."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1170834069 = messageFormatterFn((function(  ) {
  return function(d) { return "This alert is visible to everyone."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a272368871 = messageFormatterFn((function(  ) {
  return function(d) { return "Back to alerts"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a970569671 = messageFormatterFn((function(  ) {
  return function(d) { return "Back to connected alerts"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1852586489 = messageFormatterFn((function(  ) {
  return function(d) { return "Below"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b526630135 = messageFormatterFn((function(  ) {
  return function(d) { return "Below or equal to"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a85533770 = messageFormatterFn((function(  ) {
  return function(d) { return "Change owner"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1987145295 = messageFormatterFn((function(  ) {
  return function(d) { return "Change owner"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1851666224 = messageFormatterFn((function(  ) {
  return function(d) { return "Choose an owner for the alert \"" + d.alertSubject + "\""; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1998425724 = messageFormatterFn((function(  ) {
  return function(d) { return "Close"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a33435572 = messageFormatterFn((function(  ) {
  return function(d) { return "True"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1714687687 = messageFormatterFn((function(  ) {
  return function(d) { return "False"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1330692492 = messageFormatterFn((function(  ) {
  return function(d) { return "Failed"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1266007411 = messageFormatterFn((function(  ) {
  return function(d) { return "—"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1924727078 = messageFormatterFn((function(  ) {
  return function(d) { return "Custom (" + d.viewName + ")"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b605524899 = messageFormatterFn((function(  ) {
  return function(d) { return "Delete"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a997798026 = messageFormatterFn((function(  ) {
  return function(d) { return "Details"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1499363825 = messageFormatterFn((function(  ) {
  return function(d) { return "Activity"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b976535522 = messageFormatterFn((function(  ) {
  return function(d) { return "Condition"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1937685151 = messageFormatterFn((function(  ) {
  return function(d) { return "Current status"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b745794774 = messageFormatterFn((function(  ) {
  return function(d) { return "Email sent"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1186704078 = messageFormatterFn((function(  ) {
  return function(d) { return "Frequency"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b189848569 = messageFormatterFn((function(  ) {
  return function(d) { return "Last checked"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a291891063 = messageFormatterFn((function(  ) {
  return function(d) { return "Last triggered"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b469411980 = messageFormatterFn((function(  ) {
  return function(d) { return "Measure"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b497849749 = messageFormatterFn((function(  ) {
  return function(d) { return "Notification"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a751348324 = messageFormatterFn((function(  ) {
  return function(d) { return "Operator"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1372311991 = messageFormatterFn((function(  ) {
  return function(d) { return "Owner"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1857550458 = messageFormatterFn((function(  ) {
  return function(d) { return "Recipients"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1757991807 = messageFormatterFn((function(  ) {
  return function(d) { return "Threshold"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b134141275 = messageFormatterFn((function(  ) {
  return function(d) { return "View"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1279823364 = messageFormatterFn((function(  ) {
  return function(d) { return "Alert Overview"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a800557884 = messageFormatterFn((function(  ) {
  return function(d) { return "Edit"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b481213481 = messageFormatterFn((function(  ) {
  return function(d) { return "Equal to"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b938507955 = messageFormatterFn((function(  ) {
  return function(d) { return "Custom (autosaved)"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1593938012 = messageFormatterFn((function(  ) {
  return function(d) { return "Daily at most"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1819372658 = messageFormatterFn((function(  ) {
  return function(d) { return "As frequently as possible"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a437680036 = messageFormatterFn((function(  ) {
  return function(d) { return "Hourly at most"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1209439668 = messageFormatterFn((function(  ) {
  return function(d) { return "Once, the first time it's true"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b985608301 = messageFormatterFn((function(  ) {
  return function(d) { return d.interval + " minutes"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a60398004 = messageFormatterFn((function(  ) {
  return function(d) { return "Weekly at most"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b286260071 = messageFormatterFn((function(  ) {
  return function(d) { return "Invalid on "; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1547226892 = messageFormatterFn((function(  ) {
  return function(d) { return "Invalid"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a315223849 = messageFormatterFn((function(  ) {
  return function(d) { return "Last triggered " + d.lastTriggeredTime; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1358107998 = messageFormatterFn((function(  ) {
  return function(d) { return "Learn more"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1898984149 = messageFormatterFn((function(  ) {
  return function(d) { return "Alerts"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1796187200 = messageFormatterFn((function(  ) {
  return function(d) { return "Retry"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b668102034 = messageFormatterFn((function(  ) {
  return function(d) { return "Check your internet connection and retry"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a874320269 = messageFormatterFn((function(  ) {
  return function(d) { return "Unable to load alerts"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1093110127 = messageFormatterFn((function(  ) {
  return function(d) { return "Loading alerts..."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1533386505 = messageFormatterFn((function(  ) {
  return function(d) { return "Never"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1536579079 = messageFormatterFn((function(  ) {
  return function(d) { return "Never triggered"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1815470685 = messageFormatterFn((function(  ) {
  return function(d) { return "alert bell symbol"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1877719681 = messageFormatterFn((function(  ) {
  return function(d) { return "Create"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1742953853 = messageFormatterFn((function(  ) {
  return function(d) { return "Create an alert and we'll notify you when your data meets those conditions."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1153162538 = messageFormatterFn((function(  ) {
  return function(d) { return "Learn more."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1134737951 = messageFormatterFn((function(  ) {
  return function(d) { return "Select a numeric axis of a chart. Then click Create."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2012711288 = messageFormatterFn((function(  ) {
  return function(d) { return "No alerts on this view"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a836390241 = messageFormatterFn((function(  ) {
  return function(d) { return d.recipientCount + " recipients"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1237097652 = messageFormatterFn((function(  ) {
  return function(d) { return d.recipientCount + " recipient"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1086072710 = messageFormatterFn((function(  ) {
  return function(d) { return "Original"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1025518958 = messageFormatterFn((function(  ) {
  return function(d) { return "Remove me"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b966778017 = messageFormatterFn((function(  ) {
  return function(d) { return "Resume"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2119178368 = messageFormatterFn((function(  ) {
  return function(d) { return "Search"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1332896963 = messageFormatterFn((function(  ) {
  return function(d) { return "Suspended on "; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1360711504 = messageFormatterFn((function(  ) {
  return function(d) { return "Suspended"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1190955898 = messageFormatterFn((function(  ) {
  return function(d) { return "Data space"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1153348968 = messageFormatterFn((function(  ) {
  return function(d) { return "Segment name"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1732413087 = messageFormatterFn((function(  ) {
  return function(d) { return "Create Segment for Data Cloud"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1513487135 = messageFormatterFn((function(  ) {
  return function(d) { return "Description (optional) \nAutogenerated with rule details so users can view them in Data Cloud. Revise as needed."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a798019012 = messageFormatterFn((function(  ) {
  return function(d) { return "Segment on"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1930669531 = messageFormatterFn((function(  ) {
  return function(d) { return "Create a segment of your data to send to Data Cloud so that others can review, refine, and activate it."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b260722436 = messageFormatterFn((function(  ) {
  return function(d) { return "Cancel"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b204677436 = messageFormatterFn((function(  ) {
  return function(d) { return "Rules"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a198078394 = messageFormatterFn((function(  ) {
  return function(d) { return "Derived from a data source filter"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1380994337 = messageFormatterFn((function(  ) {
  return function(d) { return "Derived from selected marks on the viz"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2143406490 = messageFormatterFn((function(  ) {
  return function(d) { return "Derived from an enabled context filter"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1686916789 = messageFormatterFn((function(  ) {
  return function(d) { return "Create"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1000955472 = messageFormatterFn((function(  ) {
  return function(d) { return "Publish schedule"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1422695507 = messageFormatterFn((function(  ) {
  return function(d) { return "Publish type"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a31783367 = messageFormatterFn((function(  ) {
  return function(d) { return d.percentParam + " of " + d.wholeParam; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b923799728 = messageFormatterFn((function(  ) {
  return function(d) { return "Segment population"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1325576977 = messageFormatterFn((function(  ) {
  return function(d) { return "You can't edit segment rules after creation."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1592257067 = messageFormatterFn((function(  ) {
  return function(d) { return "The Segment Name cannot be empty."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b263951814 = messageFormatterFn((function(  ) {
  return function(d) { return "The Segment Name cannot contain a space."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a470170523 = messageFormatterFn((function(  ) {
  return function(d) { return "The Segment Name cannot end with an underscore."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b664648059 = messageFormatterFn((function(  ) {
  return function(d) { return "The Segment Name can only contain alphanumeric characters and underscores."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b559577638 = messageFormatterFn((function(  ) {
  return function(d) { return "The Segment Name must start with a letter."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1636350588 = messageFormatterFn((function(  ) {
  return function(d) { return "The Segment Name cannot contain consecutive underscores."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1628080908 = messageFormatterFn((function(  ) {
  return function(d) { return "No rules for this segment because there are no filters applied to the viz."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1693995224 = messageFormatterFn((function(  ) {
  return function(d) { return "Learn more"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2136654723 = messageFormatterFn((function(  ) {
  return function(d) { return "Loading"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b69945194 = messageFormatterFn((function( plural, pluralFuncs ) {
  return function(d) { return "is made of " + plural(d.tableCount, 0, pluralFuncs.en, { one: function() { return "1 table";}, other: function() { return d.formattedTableCount + " tables";} }) + "."; }
})(messageFormat.plural, {"en": Globalize("en").pluralGenerator()}), Globalize("en").pluralGenerator({}));
Globalize.a1013902916 = messageFormatterFn((function(  ) {
  return function(d) { return "is made of multiple tables."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a56927856 = messageFormatterFn((function(  ) {
  return function(d) { return "Logical Table: "; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a874779015 = messageFormatterFn((function(  ) {
  return function(d) { return "Physical Tables: "; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1343236990 = messageFormatterFn((function(  ) {
  return function(d) { return "Unioned Tables: "; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1904123450 = messageFormatterFn((function(  ) {
  return function(d) { return "(" + d.truncatedAmount + " More)"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1205499675 = messageFormatterFn((function(  ) {
  return function(d) { return "Many"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b272305029 = messageFormatterFn((function( plural, pluralFuncs ) {
  return function(d) { return "Double-click this logical table to see its physical " + plural(d.tableCount, 0, pluralFuncs.en, { one: function() { return "table";}, other: function() { return "tables";} }) + "."; }
})(messageFormat.plural, {"en": Globalize("en").pluralGenerator()}), Globalize("en").pluralGenerator({}));
Globalize.b1358866793 = messageFormatterFn((function( plural, pluralFuncs ) {
  return function(d) { return plural(d.tableCount, 0, pluralFuncs.en, { one: function() { return "1 physical table defines";}, other: function() { return d.formattedTableCount + " physical tables define";} }) + " the logical table "; }
})(messageFormat.plural, {"en": Globalize("en").pluralGenerator()}), Globalize("en").pluralGenerator({}));
Globalize.a498753408 = messageFormatterFn((function(  ) {
  return function(d) { return "Learn more"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a724399091 = messageFormatterFn((function(  ) {
  return function(d) { return " about the difference between physical and logical tables."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a232775331 = messageFormatterFn((function(  ) {
  return function(d) { return "Relationship: "; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1454772345 = messageFormatterFn((function(  ) {
  return function(d) { return d.name1 + " to " + d.name2; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b656820583 = messageFormatterFn((function(  ) {
  return function(d) { return "Cardinality: "; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a294635148 = messageFormatterFn((function(  ) {
  return function(d) { return "(default)"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2079978387 = messageFormatterFn((function(  ) {
  return function(d) { return "(detected)"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a752847439 = messageFormatterFn((function(  ) {
  return function(d) { return "Related Fields: "; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1285708238 = messageFormatterFn((function(  ) {
  return function(d) { return "Drag tables here"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b510230133 = messageFormatterFn((function(  ) {
  return function(d) { return "Drag tables here to create a data model"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2112857007 = messageFormatterFn((function(  ) {
  return function(d) { return "Need more data?"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1754650105 = messageFormatterFn((function(  ) {
  return function(d) { return "Need help adding data? "; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a523056616 = messageFormatterFn((function(  ) {
  return function(d) { return "Drag tables here to relate them."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a550996757 = messageFormatterFn((function(  ) {
  return function(d) { return "Learn more"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a666721833 = messageFormatterFn((function(  ) {
  return function(d) { return "One to One"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a133677070 = messageFormatterFn((function(  ) {
  return function(d) { return "One to Many"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1949106110 = messageFormatterFn((function(  ) {
  return function(d) { return "Many to One"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a647389461 = messageFormatterFn((function(  ) {
  return function(d) { return "Many to Many"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a172052352 = messageFormatterFn((function(  ) {
  return function(d) { return "Union"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b279329560 = messageFormatterFn((function(  ) {
  return function(d) { return "New Base Table"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b74343927 = messageFormatterFn((function(  ) {
  return function(d) { return "Relate tables"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1572382489 = messageFormatterFn((function(  ) {
  return function(d) { return "Add a base table"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a422032007 = messageFormatterFn((function(  ) {
  return function(d) { return "Drag tables near "; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1891497223 = messageFormatterFn((function(  ) {
  return function(d) { return " to relate them"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a77183176 = messageFormatterFn((function(  ) {
  return function(d) { return "Drag out additional base tables for multi-fact analysis."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b637658151 = messageFormatterFn((function(  ) {
  return function(d) { return "Relations:"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b604284057 = messageFormatterFn((function(  ) {
  return function(d) { return "Unrelated Tables"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1837703340 = messageFormatterFn((function(  ) {
  return function(d) { return "No error."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1906527895 = messageFormatterFn((function(  ) {
  return function(d) { return "This relationship has no matching fields."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a98531722 = messageFormatterFn((function(  ) {
  return function(d) { return " Edit the relationship to select matching fields."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a593456401 = messageFormatterFn((function(  ) {
  return function(d) { return "Type mismatch between %1 (%2) and %3 (%4)."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2042296212 = messageFormatterFn((function(  ) {
  return function(d) { return "This relationship references unknown field %1."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1813080307 = messageFormatterFn((function(  ) {
  return function(d) { return " Edit the relationship to select a valid field."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a108526753 = messageFormatterFn((function(  ) {
  return function(d) { return "There is an error in one of the relationship links."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a431325322 = messageFormatterFn((function(  ) {
  return function(d) { return "There is an error in one of the inputs to the relationship."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a914417301 = messageFormatterFn((function(  ) {
  return function(d) { return "The relationship field %1 must only appear in a single relationship comparison."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1661120372 = messageFormatterFn((function(  ) {
  return function(d) { return "This relationship type is not supported."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1933498683 = messageFormatterFn((function(  ) {
  return function(d) { return "Relationship clause contains an invalid calculation."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1108516486 = messageFormatterFn((function(  ) {
  return function(d) { return "The related field %1 must be a spatial column to use a spatial comparison."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a431275721 = messageFormatterFn((function(  ) {
  return function(d) { return "The related field %1 is a spatial column and must use a spatial comparison."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a859626263 = messageFormatterFn((function(  ) {
  return function(d) { return "The relationship uses functionality not supported by extracts stored as 'Physical Tables.'"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a195523095 = messageFormatterFn((function(  ) {
  return function(d) { return "Unknown relationship error."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1919208879 = messageFormatterFn((function(  ) {
  return function(d) { return "No matching fields"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1067787893 = messageFormatterFn((function(  ) {
  return function(d) { return "Data type mismatch"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a591426790 = messageFormatterFn((function(  ) {
  return function(d) { return "Unknown field"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a608596583 = messageFormatterFn((function(  ) {
  return function(d) { return "Relationship error"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1177411580 = messageFormatterFn((function(  ) {
  return function(d) { return "Input table error"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b619525669 = messageFormatterFn((function(  ) {
  return function(d) { return "Duplicate filed in use"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b641368134 = messageFormatterFn((function(  ) {
  return function(d) { return "Join error"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2125562421 = messageFormatterFn((function(  ) {
  return function(d) { return "Calculation error"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a163551092 = messageFormatterFn((function(  ) {
  return function(d) { return "Spatial field required"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a320934211 = messageFormatterFn((function(  ) {
  return function(d) { return "Spatial operator required"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1734006749 = messageFormatterFn((function(  ) {
  return function(d) { return "Unsupported extract type"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b872111779 = messageFormatterFn((function(  ) {
  return function(d) { return "Unknown error"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1626622132 = messageFormatterFn((function(  ) {
  return function(d) { return "Relationship"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1012425310 = messageFormatterFn((function(  ) {
  return function(d) { return "Related Fields"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1114245355 = messageFormatterFn((function(  ) {
  return function(d) { return "Relationships"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b415315055 = messageFormatterFn((function(  ) {
  return function(d) { return "more"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1059199099 = messageFormatterFn((function(  ) {
  return function(d) { return "Object graph validation warnings"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b943862966 = messageFormatterFn((function( plural, pluralFuncs ) {
  return function(d) { return d.failureCountString + " " + plural(d.failureCount, 0, pluralFuncs.en, { one: function() { return "Alert";}, other: function() { return "Alerts";} }); }
})(messageFormat.plural, {"en": Globalize("en").pluralGenerator()}), Globalize("en").pluralGenerator({}));
Globalize.a1925091770 = messageFormatterFn((function(  ) {
  return function(d) { return "This table has an invalid join."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b207274013 = messageFormatterFn((function(  ) {
  return function(d) { return "This table contains invalid physical tables or invalid joins."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1592277764 = messageFormatterFn((function(  ) {
  return function(d) { return " Please double-click this table to fix these errors."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b971270537 = messageFormatterFn((function(  ) {
  return function(d) { return "This object has an invalid analytics script."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a787063676 = messageFormatterFn((function(  ) {
  return function(d) { return "This object has no analytics connection."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a431875898 = messageFormatterFn((function(  ) {
  return function(d) { return "This object has an issue."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b710447220 = messageFormatterFn((function(  ) {
  return function(d) { return "Tableau can't find or recognize this table."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a260599341 = messageFormatterFn((function(  ) {
  return function(d) { return " Remove or replace the table."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a751698601 = messageFormatterFn((function(  ) {
  return function(d) { return "Tableau can't find or recognize this stored procedure."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2093361866 = messageFormatterFn((function(  ) {
  return function(d) { return " Remove or replace the stored procedure."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a685945530 = messageFormatterFn((function(  ) {
  return function(d) { return "Tableau doesn't recognize this table because the underlying data has changed."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b890277 = messageFormatterFn((function(  ) {
  return function(d) { return " Remove or replace table."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a858576862 = messageFormatterFn((function(  ) {
  return function(d) { return "Tableau can't find or recognize one or more tables in this union."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b526557825 = messageFormatterFn((function(  ) {
  return function(d) { return " Remove the union or replace tables."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a96404338 = messageFormatterFn((function(  ) {
  return function(d) { return "Tableau found no tables to union based on your pattern."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2142789869 = messageFormatterFn((function(  ) {
  return function(d) { return " Modify the pattern and try again."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a718834787 = messageFormatterFn((function(  ) {
  return function(d) { return "Tableau can't execute this Custom SQL Query."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b90093692 = messageFormatterFn((function(  ) {
  return function(d) { return " Modify the query and try again."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1043450755 = messageFormatterFn((function(  ) {
  return function(d) { return "All tables in a data source must be related."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a856028836 = messageFormatterFn((function(  ) {
  return function(d) { return " Relate all the tables in this model or create a separate data source for unrelated tables."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1663898802 = messageFormatterFn((function(  ) {
  return function(d) { return "There can't be more than one path between two tables."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a422328559 = messageFormatterFn((function(  ) {
  return function(d) { return " Update the data model to remove any cycles."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1980972272 = messageFormatterFn((function(  ) {
  return function(d) { return "Unrelated Tables"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1806833787 = messageFormatterFn((function(  ) {
  return function(d) { return "Cyclical Relationships"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b737340522 = messageFormatterFn((function(  ) {
  return function(d) { return "Unknown Graph Issue"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1218242903 = messageFormatterFn((function(  ) {
  return function(d) { return "The graph has an issue."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1706851742 = messageFormatterFn((function(  ) {
  return function(d) { return "Visually Separate Unrelated Tables"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1145693387 = messageFormatterFn((function(  ) {
  return function(d) { return "A data model with multiple base tables creates a data source where some fields aren't directly related."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a56268345 = messageFormatterFn((function(  ) {
  return function(d) { return "Multiple base tables"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1936330980 = messageFormatterFn((function(  ) {
  return function(d) { return "Validation failure"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1209284003 = messageFormatterFn((function(  ) {
  return function(d) { return "Empty Data Source"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1216500532 = messageFormatterFn((function(  ) {
  return function(d) { return "This published data source is read-only"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1843257803 = messageFormatterFn((function(  ) {
  return function(d) { return "Name: " + d.name; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1657225848 = messageFormatterFn((function(  ) {
  return function(d) { return "Live / Last extract: " + d.value; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a887053573 = messageFormatterFn((function(  ) {
  return function(d) { return "Project: " + d.project; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a328307534 = messageFormatterFn((function(  ) {
  return function(d) { return "Connects To: " + d.database; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a211926623 = messageFormatterFn((function(  ) {
  return function(d) { return "Owner: " + d.name; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a601030515 = messageFormatterFn((function(  ) {
  return function(d) { return "Modified: " + d.time; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b593941964 = messageFormatterFn((function(  ) {
  return function(d) { return "Views: All: " + d.count; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1244774373 = messageFormatterFn((function(  ) {
  return function(d) { return "Workbooks: " + d.count; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1990772013 = messageFormatterFn((function(  ) {
  return function(d) { return "Server: " + d.url; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1886869336 = messageFormatterFn((function(  ) {
  return function(d) { return "Description: " + d.text; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b449613626 = messageFormatterFn((function(  ) {
  return function(d) { return d.upstream + " to " + d.downstream + "."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b128470323 = messageFormatterFn((function(  ) {
  return function(d) { return d.upstream + " to " + d.downstream + ". Use Left and Right arrows to navigate to upstream and downstream tables. Use Up and Down arrows to select other relationships in the bundle. Press Space or Enter to select the relationship."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a765199749 = messageFormatterFn((function(  ) {
  return function(d) { return "Base: " + d.caption + "."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1562445038 = messageFormatterFn((function(  ) {
  return function(d) { return "Base: " + d.caption + ". Use arrow keys to navigate to downstream relationships and related tables. Press Space or Enter to select the table and access the context menu. Press Escape to return to the table."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b326284364 = messageFormatterFn((function(  ) {
  return function(d) { return "Base: " + d.caption + " collapsed. Press Space or Enter to expand."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a99024918 = messageFormatterFn((function(  ) {
  return function(d) { return d.caption + "."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1311268221 = messageFormatterFn((function(  ) {
  return function(d) { return d.caption + ". Use Left and Right arrows to navigate to outgoing relationships. Use Up and Down arrows to navigate to tables not directly related. Use Left and Right arrows to navigate to upstream and downstream tables. Press Space or Enter to select the table and access the context menu. Press Escape to return to the table."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1269805617 = messageFormatterFn((function(  ) {
  return function(d) { return d.caption + ", shared table downstream of " + d.baseTableList + "."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b574170686 = messageFormatterFn((function(  ) {
  return function(d) { return d.caption + ", shared table downstream of " + d.baseTableList + ". Use Left and Right arrows to navigate to outgoing relationships. Use Up and Down arrows to navigate to tables not directly related. Use Left and Right arrows to navigate to upstream and downstream tables. Press Space or Enter to select the table and access the context menu. Press Escape to return to the table."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2059793916 = messageFormatterFn((function(  ) {
  return function(d) { return d.caption + ", downstream of " + d.baseTableList + ". No further downstream tables to explore. Use Left arrow to move back upstream."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a543898259 = messageFormatterFn((function(  ) {
  return function(d) { return "Dashboard"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2085309850 = messageFormatterFn((function(  ) {
  return function(d) { return "Viz"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b485316348 = messageFormatterFn((function(  ) {
  return function(d) { return "Mark"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1475842568 = messageFormatterFn((function(  ) {
  return function(d) { return "This " + d.paneTypeText; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1369971142 = messageFormatterFn((function(  ) {
  return function(d) { return "These marks"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a557602620 = messageFormatterFn((function(  ) {
  return function(d) { return "Enter a description that helps users understand this dashboard"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b870399537 = messageFormatterFn((function(  ) {
  return function(d) { return "Enter a description that helps users understand this viz"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1804338737 = messageFormatterFn((function(  ) {
  return function(d) { return d.paneTypeText + " description"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a583074855 = messageFormatterFn((function(  ) {
  return function(d) { return d.paneTypeText + " Details"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2035030852 = messageFormatterFn((function(  ) {
  return function(d) { return "Data in This " + d.paneTypeText; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2103485898 = messageFormatterFn((function(  ) {
  return function(d) { return "Data in These Marks"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b273704696 = messageFormatterFn((function(  ) {
  return function(d) { return "Data Summary"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1163070811 = messageFormatterFn((function(  ) {
  return function(d) { return "Has 1 data point"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a903201855 = messageFormatterFn((function(  ) {
  return function(d) { return "Has " + d.includedDataPointCount + " data points"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1639382351 = messageFormatterFn((function(  ) {
  return function(d) { return "Has 1 record"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b256589531 = messageFormatterFn((function(  ) {
  return function(d) { return "Has " + d.recordCount + " records"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b451629833 = messageFormatterFn((function(  ) {
  return function(d) { return "Have 1 record"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1287720469 = messageFormatterFn((function(  ) {
  return function(d) { return "Have " + d.recordCount + " records"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1282068138 = messageFormatterFn((function(  ) {
  return function(d) { return "Is sorted in " + d.sortDirection + " order based on "; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a547977980 = messageFormatterFn((function(  ) {
  return function(d) { return d.fieldName; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a827996443 = messageFormatterFn((function(  ) {
  return function(d) { return d.shelfType; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a315559871 = messageFormatterFn((function(  ) {
  return function(d) { return "sorting"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1945897628 = messageFormatterFn((function(  ) {
  return function(d) { return "Columns"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b329084656 = messageFormatterFn((function(  ) {
  return function(d) { return "Rows"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1433151009 = messageFormatterFn((function(  ) {
  return function(d) { return "Is ranked " + d.currentIndex + " out of " + d.totalNumber + " based on "; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1459016629 = messageFormatterFn((function(  ) {
  return function(d) { return "Applied Filters"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1717956345 = messageFormatterFn((function(  ) {
  return function(d) { return "No filters apply to this viz"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1451818705 = messageFormatterFn((function(  ) {
  return function(d) { return "No filters apply to this mark"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2051004919 = messageFormatterFn((function(  ) {
  return function(d) { return "No filters apply to these marks"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a218289039 = messageFormatterFn((function(  ) {
  return function(d) { return "No data available"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1324206208 = messageFormatterFn((function(  ) {
  return function(d) { return "Total data in " + d.thisItem + " is prefiltered to "; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b242645710 = messageFormatterFn((function(  ) {
  return function(d) { return "an ascending"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1141413126 = messageFormatterFn((function(  ) {
  return function(d) { return "a descending"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1162270032 = messageFormatterFn((function(  ) {
  return function(d) { return "Couldn't update the description. Try again."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a939031101 = messageFormatterFn((function(  ) {
  return function(d) { return "Additional resources"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b412797062 = messageFormatterFn((function(  ) {
  return function(d) { return "Add link"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1232673011 = messageFormatterFn((function(  ) {
  return function(d) { return "Label"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2086962770 = messageFormatterFn((function(  ) {
  return function(d) { return "URL"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b711838463 = messageFormatterFn((function(  ) {
  return function(d) { return "Edit"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1004648955 = messageFormatterFn((function(  ) {
  return function(d) { return "Remove"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b245132121 = messageFormatterFn((function(  ) {
  return function(d) { return "Add or edit link"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b419315829 = messageFormatterFn((function(  ) {
  return function(d) { return "OK"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1028828506 = messageFormatterFn((function(  ) {
  return function(d) { return "Data Guide"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1071281745 = messageFormatterFn((function(  ) {
  return function(d) { return "Explain Data"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b501852137 = messageFormatterFn((function(  ) {
  return function(d) { return "Please enter a valid URL"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a593660513 = messageFormatterFn((function(  ) {
  return function(d) { return "published on " + d.date; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1913738660 = messageFormatterFn((function(  ) {
  return function(d) { return "Published by "; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a298188787 = messageFormatterFn((function(  ) {
  return function(d) { return "number"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b447258920 = messageFormatterFn((function(  ) {
  return function(d) { return "date"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b259348475 = messageFormatterFn((function(  ) {
  return function(d) { return "date & time"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b315329861 = messageFormatterFn((function(  ) {
  return function(d) { return "string"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1062471188 = messageFormatterFn((function(  ) {
  return function(d) { return "boolean"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b981376131 = messageFormatterFn((function(  ) {
  return function(d) { return "geographic"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a359161672 = messageFormatterFn((function(  ) {
  return function(d) { return "attribute"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b290761418 = messageFormatterFn((function(  ) {
  return function(d) { return "varying attribute"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1941370544 = messageFormatterFn((function(  ) {
  return function(d) { return "level"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1389572004 = messageFormatterFn((function(  ) {
  return function(d) { return "numeric bin"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a885684977 = messageFormatterFn((function(  ) {
  return function(d) { return "group"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b970289074 = messageFormatterFn((function(  ) {
  return function(d) { return "set"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a725428028 = messageFormatterFn((function(  ) {
  return function(d) { return "other"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a746984850 = messageFormatterFn((function(  ) {
  return function(d) { return "Learn more"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1253770870 = messageFormatterFn((function(  ) {
  return function(d) { return "Help"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1856479432 = messageFormatterFn((function(  ) {
  return function(d) { return "Go back"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2142330863 = messageFormatterFn((function(  ) {
  return function(d) { return "No Outlier Marks Found"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a67270447 = messageFormatterFn((function(  ) {
  return function(d) { return "Detecting Outliers..."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b579706475 = messageFormatterFn((function(  ) {
  return function(d) { return "Detected Outliers"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b304147845 = messageFormatterFn((function(  ) {
  return function(d) { return "Detected Outlier (1)"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a145163922 = messageFormatterFn((function(  ) {
  return function(d) { return "Detected Outliers (" + d.NumberOfOutliers + ")"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a843302027 = messageFormatterFn((function(  ) {
  return function(d) { return "Tableau detected these marks as unusual compared to other marks in the same viz."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b581298594 = messageFormatterFn((function(  ) {
  return function(d) { return "Tableau detected these marks as unusual compared to other marks in the selected viz."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b188588484 = messageFormatterFn((function(  ) {
  return function(d) { return d.MeasureName + ": " + d.MeasureValue; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b207813796 = messageFormatterFn((function(  ) {
  return function(d) { return "High"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1249743568 = messageFormatterFn((function(  ) {
  return function(d) { return "Low"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1414222060 = messageFormatterFn((function(  ) {
  return function(d) { return "Do you want to see possible explanations for these outliers? To turn on explanations, contact: "; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1390357783 = messageFormatterFn((function(  ) {
  return function(d) { return "Show more"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a862610481 = messageFormatterFn((function(  ) {
  return function(d) { return "Explanations"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1366842969 = messageFormatterFn((function(  ) {
  return function(d) { return "Tableau detected these notable data changes based on historical trends."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1864657466 = messageFormatterFn((function(  ) {
  return function(d) { return "To ask Tableau to monitor data changes for this view, or to see monitoring status, go to this view's workbook page."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1859159755 = messageFormatterFn((function(  ) {
  return function(d) { return "Notable Data Changes"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1090966872 = messageFormatterFn((function(  ) {
  return function(d) { return "Notable Data Changes (" + d.NumberOfSignals + ")"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b915283797 = messageFormatterFn((function(  ) {
  return function(d) { return "Detecting Data Changes..."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a285363040 = messageFormatterFn((function(  ) {
  return function(d) { return d.aggTargetMeasureLabel + " is lower than predicted"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a215574532 = messageFormatterFn((function(  ) {
  return function(d) { return d.aggTargetMeasureLabel + " is higher than predicted"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1245840286 = messageFormatterFn((function(  ) {
  return function(d) { return "Value is unexpected"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1063727159 = messageFormatterFn((function(  ) {
  return function(d) { return "The filter and parameter settings have changed. No data changes are available for the current settings."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b476872311 = messageFormatterFn((function(  ) {
  return function(d) { return "No unexpected data change has been found in this view in the past 24 hours."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a106663703 = messageFormatterFn((function(  ) {
  return function(d) { return "All: Last " + d.numDays + " days"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1087823099 = messageFormatterFn((function(  ) {
  return function(d) { return "Latest: Last " + d.numHours + " hours"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a102243265 = messageFormatterFn((function(  ) {
  return function(d) { return "Lower than predicted: found " + d.timeStamp; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1767363227 = messageFormatterFn((function(  ) {
  return function(d) { return "Higher than predicted: found " + d.timeStamp; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1022809182 = messageFormatterFn((function(  ) {
  return function(d) { return d.aggTargetMeasureLabel + ": lower than predicted"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a953020674 = messageFormatterFn((function(  ) {
  return function(d) { return d.aggTargetMeasureLabel + ": higher than predicted"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2100912864 = messageFormatterFn((function(  ) {
  return function(d) { return "Value is unexpected"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a512785395 = messageFormatterFn((function(  ) {
  return function(d) { return "sum of " + d.measureCaption; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a457492193 = messageFormatterFn((function(  ) {
  return function(d) { return "average of " + d.measureCaption; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1614153809 = messageFormatterFn((function(  ) {
  return function(d) { return "count of " + d.measureCaption; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2099540849 = messageFormatterFn((function(  ) {
  return function(d) { return "distinct count of " + d.measureCaption; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a43588700 = messageFormatterFn((function(  ) {
  return function(d) { return "median of " + d.measureCaption; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a677764862 = messageFormatterFn((function(  ) {
  return function(d) { return "maximum of " + d.measureCaption; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b570284976 = messageFormatterFn((function(  ) {
  return function(d) { return "minimum of " + d.measureCaption; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a861606234 = messageFormatterFn((function(  ) {
  return function(d) { return "standard deviation of " + d.measureCaption; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a837286755 = messageFormatterFn((function(  ) {
  return function(d) { return "standard deviation (from population) of " + d.measureCaption; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1468644459 = messageFormatterFn((function(  ) {
  return function(d) { return "variance of " + d.measureCaption; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1005199160 = messageFormatterFn((function(  ) {
  return function(d) { return "variance (from population) of " + d.measureCaption; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1800591233 = messageFormatterFn((function(  ) {
  return function(d) { return d.N + "st/nd/rd/th percentile of " + d.measureCaption; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1713020197 = messageFormatterFn((function(  ) {
  return function(d) { return "Sum of " + d.measureCaption; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b442067155 = messageFormatterFn((function(  ) {
  return function(d) { return "Average of " + d.measureCaption; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1274444191 = messageFormatterFn((function(  ) {
  return function(d) { return "Count of " + d.measureCaption; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1586466659 = messageFormatterFn((function(  ) {
  return function(d) { return "Distinct count of " + d.measureCaption; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1877633326 = messageFormatterFn((function(  ) {
  return function(d) { return "Median of " + d.measureCaption; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1870251280 = messageFormatterFn((function(  ) {
  return function(d) { return "Maximum of " + d.measureCaption; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b639381922 = messageFormatterFn((function(  ) {
  return function(d) { return "Minimum of " + d.measureCaption; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a9778964 = messageFormatterFn((function(  ) {
  return function(d) { return "Standard deviation of " + d.measureCaption; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1492180331 = messageFormatterFn((function(  ) {
  return function(d) { return "Standard deviation (from population) of " + d.measureCaption; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1423434233 = messageFormatterFn((function(  ) {
  return function(d) { return "Variance of " + d.measureCaption; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1968695578 = messageFormatterFn((function(  ) {
  return function(d) { return "Variance (from population) of " + d.measureCaption; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a854382273 = messageFormatterFn((function(  ) {
  return function(d) { return "Was it useful to identify this data change?"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1944994211 = messageFormatterFn((function(  ) {
  return function(d) { return "Thank you for the feedback!"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1525906322 = messageFormatterFn((function(  ) {
  return function(d) { return "Your response has been recorded."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1695169351 = messageFormatterFn((function(  ) {
  return function(d) { return "Yes"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1331110985 = messageFormatterFn((function(  ) {
  return function(d) { return "No"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1781782366 = messageFormatterFn((function(  ) {
  return function(d) { return "We are having trouble retrieving this information. Please try again."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a395219392 = messageFormatterFn((function(  ) {
  return function(d) { return "Tableau found that " + d.unexpectedValue + " is notably lower than values it predicted for " + d.aggTargetMeasureLabel + "."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a325430884 = messageFormatterFn((function(  ) {
  return function(d) { return "Tableau found that " + d.unexpectedValue + " is notably higher than values it predicted for " + d.aggTargetMeasureLabel + "."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1386956458 = messageFormatterFn((function(  ) {
  return function(d) { return d.aggTargetMeasureLabel + " for " + d.dimensionNames; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b392479784 = messageFormatterFn((function(  ) {
  return function(d) { return d.truncatedHeader + "..."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2052572250 = messageFormatterFn((function(  ) {
  return function(d) { return "Data Change detected " + d.RELATIVE_TIME + ", at " + d.TIMESTAMP + " on " + d.DATE + "."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a507533143 = messageFormatterFn((function(  ) {
  return function(d) { return "Current value"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b941568844 = messageFormatterFn((function(  ) {
  return function(d) { return "Lower than predicted range"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a575381433 = messageFormatterFn((function(  ) {
  return function(d) { return "Within predicted range"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b117430690 = messageFormatterFn((function(  ) {
  return function(d) { return "Higher than predicted range"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1110823580 = messageFormatterFn((function(  ) {
  return function(d) { return "Predicted range unavailable"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1499527699 = messageFormatterFn((function(  ) {
  return function(d) { return d.MIN + " to " + d.MAX; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a357150344 = messageFormatterFn((function(  ) {
  return function(d) { return "About this predicted range."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1165214244 = messageFormatterFn((function(  ) {
  return function(d) { return "To estimate the predicted range, Tableau trains a machine learning model by automatically recording this measure’s value at each data refresh."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a178793172 = messageFormatterFn((function(  ) {
  return function(d) { return "The model may use more historical records than what is shown in the chart, up to a maximum of 960 data points."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1674672024 = messageFormatterFn((function(  ) {
  return function(d) { return "1D"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1334919839 = messageFormatterFn((function(  ) {
  return function(d) { return "7D"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1247749184 = messageFormatterFn((function(  ) {
  return function(d) { return "14D"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1948423640 = messageFormatterFn((function(  ) {
  return function(d) { return "Max"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1178143005 = messageFormatterFn((function(  ) {
  return function(d) { return "Notable data change information is no longer available."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a126657312 = messageFormatterFn((function(  ) {
  return function(d) { return "Accessibility"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b260772351 = messageFormatterFn((function(  ) {
  return function(d) { return "Edit alt text"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1744109194 = messageFormatterFn((function(  ) {
  return function(d) { return "Tableau Pulse Metrics"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1540353729 = messageFormatterFn((function(  ) {
  return function(d) { return "Ascending order"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b730750761 = messageFormatterFn((function(  ) {
  return function(d) { return "Descending order"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a62611803 = messageFormatterFn((function(  ) {
  return function(d) { return "Close Data Guide"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b790432351 = messageFormatterFn((function(  ) {
  return function(d) { return "Sort Fields:"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1203608173 = messageFormatterFn((function(  ) {
  return function(d) { return "Show aliases"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b605992246 = messageFormatterFn((function(  ) {
  return function(d) { return "Show hidden fields"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2123238333 = messageFormatterFn((function(  ) {
  return function(d) { return "rows"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1568205289 = messageFormatterFn((function(  ) {
  return function(d) { return "Data source order"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1342062568 = messageFormatterFn((function(  ) {
  return function(d) { return "A to Z ascending"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b601400916 = messageFormatterFn((function(  ) {
  return function(d) { return "Z to A descending"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2006498169 = messageFormatterFn((function(  ) {
  return function(d) { return "A to Z ascending per table"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b33697475 = messageFormatterFn((function(  ) {
  return function(d) { return "Z to A descending per table"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1207975537 = messageFormatterFn((function(  ) {
  return function(d) { return "Modified"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a702061038 = messageFormatterFn((function(  ) {
  return function(d) { return "Group"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1794264026 = messageFormatterFn((function(  ) {
  return function(d) { return "Bin"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2104396134 = messageFormatterFn((function(  ) {
  return function(d) { return "Calculation"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a747505093 = messageFormatterFn((function(  ) {
  return function(d) { return "Can't rename the Field"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b427207708 = messageFormatterFn((function(  ) {
  return function(d) { return "A field or logical table named '" + d.fieldName + "' already exists."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2043971701 = messageFormatterFn((function(  ) {
  return function(d) { return "Name"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1311315613 = messageFormatterFn((function(  ) {
  return function(d) { return "Fields"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b168918087 = messageFormatterFn((function(  ) {
  return function(d) { return "Update Now"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1907329210 = messageFormatterFn((function(  ) {
  return function(d) { return "Update Automatically"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b505500444 = messageFormatterFn((function(  ) {
  return function(d) { return "Cancel"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a654666106 = messageFormatterFn((function(  ) {
  return function(d) { return "Logical Tables"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1810970535 = messageFormatterFn((function(  ) {
  return function(d) { return "Relationships"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a745374586 = messageFormatterFn((function(  ) {
  return function(d) { return "Cardinality"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1499976169 = messageFormatterFn((function(  ) {
  return function(d) { return d.fieldCount + " fields " + d.rowCount + " rows"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a788816655 = messageFormatterFn((function(  ) {
  return function(d) { return "One to One"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1385401556 = messageFormatterFn((function(  ) {
  return function(d) { return "One to Many"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b90552968 = messageFormatterFn((function(  ) {
  return function(d) { return "Many to One"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1418911203 = messageFormatterFn((function(  ) {
  return function(d) { return "Many to Many"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b853747940 = messageFormatterFn((function(  ) {
  return function(d) { return "Data preview unavailable"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1235108613 = messageFormatterFn((function(  ) {
  return function(d) { return "No fields available"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1655474700 = messageFormatterFn((function(  ) {
  return function(d) { return "The logical table " + d.tableName + " has " + d.numCols + " columns, which is over the preview limit of " + d.colLimit + " columns. To see the contents of this table, reduce the number of columns in the table, build a view, or use View Data."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1207131916 = messageFormatterFn((function(  ) {
  return function(d) { return "Edit Relationship"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1592222914 = messageFormatterFn((function(  ) {
  return function(d) { return "Table Details"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1690415778 = messageFormatterFn((function(  ) {
  return function(d) { return "No matches."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b171915313 = messageFormatterFn((function(  ) {
  return function(d) { return "Input Fields"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b162691516 = messageFormatterFn((function(  ) {
  return function(d) { return "Output Fields"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b420106610 = messageFormatterFn((function(  ) {
  return function(d) { return "Input Table"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a827116921 = messageFormatterFn((function(  ) {
  return function(d) { return "Output Table"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b568384954 = messageFormatterFn((function(  ) {
  return function(d) { return "Apply analytics extension table changes to generate data preview"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b76568749 = messageFormatterFn((function(  ) {
  return function(d) { return "Copy Values"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1877937160 = messageFormatterFn((function(  ) {
  return function(d) { return "Enter number of rows to display"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a413939360 = messageFormatterFn((function(  ) {
  return function(d) { return "Edit number of rows to display"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1827278180 = messageFormatterFn((function(  ) {
  return function(d) { return "Settings"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1897604457 = messageFormatterFn((function(  ) {
  return function(d) { return "Warning"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1765306103 = messageFormatterFn((function(  ) {
  return function(d) { return "Collapse data preview area"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1872579428 = messageFormatterFn((function(  ) {
  return function(d) { return "Expand data preview area"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b823992008 = messageFormatterFn((function(  ) {
  return function(d) { return "Collapse relationship details pane"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1155071013 = messageFormatterFn((function(  ) {
  return function(d) { return "Expand relationship details pane"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1379531744 = messageFormatterFn((function(  ) {
  return function(d) { return "More options for field name " + d.NAME; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1059244121 = messageFormatterFn((function(  ) {
  return function(d) { return "Type"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1705291068 = messageFormatterFn((function(  ) {
  return function(d) { return "Field Name"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1760368974 = messageFormatterFn((function(  ) {
  return function(d) { return "Physical Table"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1689296918 = messageFormatterFn((function(  ) {
  return function(d) { return "Remote Field Name"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a883662694 = messageFormatterFn((function(  ) {
  return function(d) { return "null"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2125032475 = messageFormatterFn((function(  ) {
  return function(d) { return "number"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1191725824 = messageFormatterFn((function(  ) {
  return function(d) { return "date"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1304188371 = messageFormatterFn((function(  ) {
  return function(d) { return "date & time"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1511513827 = messageFormatterFn((function(  ) {
  return function(d) { return "string"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1860050668 = messageFormatterFn((function(  ) {
  return function(d) { return "boolean"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a518629461 = messageFormatterFn((function(  ) {
  return function(d) { return "geographic"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1966104032 = messageFormatterFn((function(  ) {
  return function(d) { return "attribute"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2115085921 = messageFormatterFn((function(  ) {
  return function(d) { return "Image"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1695305890 = messageFormatterFn((function(  ) {
  return function(d) { return "varying attribute"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a337733000 = messageFormatterFn((function(  ) {
  return function(d) { return "level"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1974188084 = messageFormatterFn((function(  ) {
  return function(d) { return "numeric bin"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a406011673 = messageFormatterFn((function(  ) {
  return function(d) { return "group"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a529716518 = messageFormatterFn((function(  ) {
  return function(d) { return "set"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b878209516 = messageFormatterFn((function(  ) {
  return function(d) { return "other"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a560060940 = messageFormatterFn((function(  ) {
  return function(d) { return "Collapse"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1234925895 = messageFormatterFn((function(  ) {
  return function(d) { return "Expand"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1062512815 = messageFormatterFn((function(  ) {
  return function(d) { return "Sort, Ascending"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a455803287 = messageFormatterFn((function(  ) {
  return function(d) { return "Sort, Descending"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1350685150 = messageFormatterFn((function(  ) {
  return function(d) { return "Sort, None"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b933979678 = messageFormatterFn((function(  ) {
  return function(d) { return "Live"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1699798881 = messageFormatterFn((function(  ) {
  return function(d) { return "Extract"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1936658290 = messageFormatterFn((function(  ) {
  return function(d) { return "Connection"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b947838063 = messageFormatterFn((function(  ) {
  return function(d) { return "Filters"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1894055211 = messageFormatterFn((function(  ) {
  return function(d) { return "Add"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b872591436 = messageFormatterFn((function(  ) {
  return function(d) { return "Edit"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a659760402 = messageFormatterFn((function(  ) {
  return function(d) { return "Cross-database join"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2128579395 = messageFormatterFn((function(  ) {
  return function(d) { return "Edit"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a438503872 = messageFormatterFn((function(  ) {
  return function(d) { return "Remote join location"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a674040967 = messageFormatterFn((function(  ) {
  return function(d) { return "Default join location"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1307573579 = messageFormatterFn((function(  ) {
  return function(d) { return "Data is moved from a file-based connection to the database. This option ignores the file's size and may impact performance."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1845398108 = messageFormatterFn((function(  ) {
  return function(d) { return "Data may be moved across connections and joined in a database, or the join may occur in Tableau."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1953899241 = messageFormatterFn((function(  ) {
  return function(d) { return "Create Extract"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b493160402 = messageFormatterFn((function(  ) {
  return function(d) { return "Edit"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1594814515 = messageFormatterFn((function(  ) {
  return function(d) { return "Refresh"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1525825699 = messageFormatterFn((function(  ) {
  return function(d) { return "Extract contains subset of data. "; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b674007575 = messageFormatterFn((function(  ) {
  return function(d) { return "Extract will contain subset of data. "; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1385021981 = messageFormatterFn((function(  ) {
  return function(d) { return "Extract will be created automatically when you switch to a different data source or when you switch tabs."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b919474122 = messageFormatterFn((function(  ) {
  return function(d) { return "Extract contains all data. "; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1480938160 = messageFormatterFn((function(  ) {
  return function(d) { return "Extract will contain all data. "; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1218927872 = messageFormatterFn((function(  ) {
  return function(d) { return "Extract contains user-filtered data. "; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1810728410 = messageFormatterFn((function(  ) {
  return function(d) { return "Extract will contain user-filtered data. "; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a704027815 = messageFormatterFn((function(  ) {
  return function(d) { return "Extract contains a subset of user-filtered data. "; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1351615199 = messageFormatterFn((function(  ) {
  return function(d) { return "Extract will contain a subset of user-filtered data. "; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a411004451 = messageFormatterFn((function(  ) {
  return function(d) { return "Comma"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b368659549 = messageFormatterFn((function(  ) {
  return function(d) { return "Tab"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b483997825 = messageFormatterFn((function(  ) {
  return function(d) { return "Semicolon"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1743414292 = messageFormatterFn((function(  ) {
  return function(d) { return "Space"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a896903115 = messageFormatterFn((function(  ) {
  return function(d) { return "Vertical Bar"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1685547106 = messageFormatterFn((function(  ) {
  return function(d) { return "Other"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b235093100 = messageFormatterFn((function(  ) {
  return function(d) { return "Automatic"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a773932357 = messageFormatterFn((function(  ) {
  return function(d) { return "None"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b739467495 = messageFormatterFn((function(  ) {
  return function(d) { return "Field separator:"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b516599906 = messageFormatterFn((function(  ) {
  return function(d) { return "Text qualifier:"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1272973824 = messageFormatterFn((function(  ) {
  return function(d) { return "Character set:"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a782488060 = messageFormatterFn((function(  ) {
  return function(d) { return "Locale:"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1868788079 = messageFormatterFn((function(  ) {
  return function(d) { return "No matches."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b269370960 = messageFormatterFn((function(  ) {
  return function(d) { return "Drag table to union"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2077382616 = messageFormatterFn((function(  ) {
  return function(d) { return "Apply"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b968309828 = messageFormatterFn((function(  ) {
  return function(d) { return "OK"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b137806641 = messageFormatterFn((function(  ) {
  return function(d) { return "Connection: "; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b593092693 = messageFormatterFn((function(  ) {
  return function(d) { return "Drag tables here"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b8866189 = messageFormatterFn((function(  ) {
  return function(d) { return "Tables in union: "; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a600859031 = messageFormatterFn((function(  ) {
  return function(d) { return "You can only union data from the same connection"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2082057339 = messageFormatterFn((function(  ) {
  return function(d) { return "Tableau can't find or recognize this table. Remove or replace the table"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a655081587 = messageFormatterFn((function(  ) {
  return function(d) { return "Search"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1284325273 = messageFormatterFn((function(  ) {
  return function(d) { return "Only the first table in your selection will be added because union is not available for this database."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1973903289 = messageFormatterFn((function( plural, pluralFuncs ) {
  return function(d) { return d.ValidationFailuresCountString + " " + plural(d.ValidationFailuresCount, 0, pluralFuncs.en, { one: function() { return "Alert";}, other: function() { return "Alerts";} }); }
})(messageFormat.plural, {"en": Globalize("en").pluralGenerator()}), Globalize("en").pluralGenerator({}));
Globalize.b1314122200 = messageFormatterFn((function(  ) {
  return function(d) { return "Bell with a red circle on top of it"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1181813205 = messageFormatterFn((function(  ) {
  return function(d) { return "Data Source Can"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a645959556 = messageFormatterFn((function(  ) {
  return function(d) { return "Triangle pointing down"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a387846873 = messageFormatterFn((function(  ) {
  return function(d) { return "Edit data source name"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a596771286 = messageFormatterFn((function(  ) {
  return function(d) { return "Extract tooltip info icon"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1722519201 = messageFormatterFn((function(  ) {
  return function(d) { return "Cancel"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1130812454 = messageFormatterFn((function(  ) {
  return function(d) { return "Edit Data Source Filters"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b315487901 = messageFormatterFn((function(  ) {
  return function(d) { return "OK"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a65829648 = messageFormatterFn((function(  ) {
  return function(d) { return "Add Filter"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b58863481 = messageFormatterFn((function(  ) {
  return function(d) { return "Edit"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1944391173 = messageFormatterFn((function(  ) {
  return function(d) { return "Filter"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a131217698 = messageFormatterFn((function(  ) {
  return function(d) { return "Details"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a980959317 = messageFormatterFn((function(  ) {
  return function(d) { return "Scope"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1003309247 = messageFormatterFn((function(  ) {
  return function(d) { return "Remove"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1468263129 = messageFormatterFn((function(  ) {
  return function(d) { return d.objectCaption + " and related tables"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1032498592 = messageFormatterFn((function(  ) {
  return function(d) { return "All related tables"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1449255604 = messageFormatterFn((function(  ) {
  return function(d) { return "This filter can't be scoped to a single table because it references multiple tables."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b58137761 = messageFormatterFn((function(  ) {
  return function(d) { return "Other"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a44020938 = messageFormatterFn((function(  ) {
  return function(d) { return "Copy"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1395604282 = messageFormatterFn((function(  ) {
  return function(d) { return "Copy link address"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b251899894 = messageFormatterFn((function(  ) {
  return function(d) { return "Dismiss"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1827707147 = messageFormatterFn((function(  ) {
  return function(d) { return "Error Code: "; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b699475681 = messageFormatterFn((function(  ) {
  return function(d) { return "Unable to complete action"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a961972272 = messageFormatterFn((function(  ) {
  return function(d) { return "Copy Error Message"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b815372396 = messageFormatterFn((function(  ) {
  return function(d) { return "Go to Support"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b255654840 = messageFormatterFn((function(  ) {
  return function(d) { return "Yes"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1809682794 = messageFormatterFn((function(  ) {
  return function(d) { return "No"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b793588198 = messageFormatterFn((function(  ) {
  return function(d) { return "Error copied to clipboard"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1139677772 = messageFormatterFn((function(  ) {
  return function(d) { return "Created from: "; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1304886176 = messageFormatterFn((function(  ) {
  return function(d) { return "... + " + d.count; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1453694534 = messageFormatterFn((function(  ) {
  return function(d) { return "Warning: "; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b758717554 = messageFormatterFn((function(  ) {
  return function(d) { return "Error: "; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1315215116 = messageFormatterFn((function(  ) {
  return function(d) { return "This measure isn't related to any dimensions in the viz. If used, it won't be broken down."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1970365892 = messageFormatterFn((function(  ) {
  return function(d) { return "This dimension isn't related to any dimensions in the viz. If used, it will show all possible combinations of values with other unrelated dimensions."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b126256060 = messageFormatterFn((function(  ) {
  return function(d) { return "This dimension isn't related to any measures in the viz. If used, it won't break down measure values."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1560043511 = messageFormatterFn((function(  ) {
  return function(d) { return "Unrelated dimensions can show combinations of values in the viz that don't exist in the data. Unrelated to:"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b615843591 = messageFormatterFn((function(  ) {
  return function(d) { return "Unrelated dimensions show all possible combinations of values. Unrelated to:"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1095263809 = messageFormatterFn((function(  ) {
  return function(d) { return "This dimension can't break down unrelated measures:"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b128736767 = messageFormatterFn((function(  ) {
  return function(d) { return "This measure can't be broken down by unrelated dimensions:"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a946027273 = messageFormatterFn((function(  ) {
  return function(d) { return "This field isn't filtered by unrelated filters:"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1122421644 = messageFormatterFn((function(  ) {
  return function(d) { return "Unrelated dimensions can show combinations of values in the viz that don't exist in the data. Unrelated to:"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1539915140 = messageFormatterFn((function(  ) {
  return function(d) { return "If used, this dimension will show all possible combinations of values with unrelated dimensions:"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2062265972 = messageFormatterFn((function(  ) {
  return function(d) { return "If used, this dimension won't break down unrelated measures:"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1717442698 = messageFormatterFn((function(  ) {
  return function(d) { return "This dimension can't break down the measure " + d.measures + " in combination with the following dimensions:"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1262267863 = messageFormatterFn((function(  ) {
  return function(d) { return "This dimension can't break down the measure " + d.measures + " in combination with the following groups of dimensions:"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1041262586 = messageFormatterFn((function(  ) {
  return function(d) { return "This dimension can't break down the measures " + d.measures + " in combination with the following dimensions:"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b913083123 = messageFormatterFn((function(  ) {
  return function(d) { return "This dimension can't break down the measures " + d.measures + " in combination with the following groups of dimensions:"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1067620779 = messageFormatterFn((function(  ) {
  return function(d) { return "If used, this dimension can't break down the measure " + d.measures + " in combination with the following dimensions:"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a405620460 = messageFormatterFn((function(  ) {
  return function(d) { return "If used, this dimension can't break down the measure " + d.measures + " in combination with the following groups of dimensions:"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b849693489 = messageFormatterFn((function(  ) {
  return function(d) { return "If used, this dimension can't break down the measures " + d.measures + " in combination with the following dimensions:"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1414587736 = messageFormatterFn((function(  ) {
  return function(d) { return "If used, this dimension can't break down the measures " + d.measures + " in combination with the following groups of dimensions:"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1716847453 = messageFormatterFn((function(  ) {
  return function(d) { return "Unrelated dimensions show all possible combinations of values. This dimension is not yet related to and so is treated as unrelated to the following:"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a190429605 = messageFormatterFn((function(  ) {
  return function(d) { return "This dimension is not yet related to and so can't break down the following:"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a223471717 = messageFormatterFn((function(  ) {
  return function(d) { return "This measure is not yet related to and so can't be broken down by the following:"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1035471464 = messageFormatterFn((function(  ) {
  return function(d) { return "If used, this dimension won't yet be related to and so is treated as unrelated to the following:"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1700333424 = messageFormatterFn((function(  ) {
  return function(d) { return "If used, this dimension will not yet be related to and so won't break down the following:"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1733375536 = messageFormatterFn((function(  ) {
  return function(d) { return "If used, this measure won't yet be related to and and so won't be broken down by the following:"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1008700748 = messageFormatterFn((function(  ) {
  return function(d) { return "If used, this measure won't be broken down by unrelated dimensions:"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b202996780 = messageFormatterFn((function(  ) {
  return function(d) { return "If used, this field will be ignored by unrelated filters:"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1254380598 = messageFormatterFn((function(  ) {
  return function(d) { return "If used, this measure can only be broken down by one of these dimensions at a time:"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b96242454 = messageFormatterFn((function(  ) {
  return function(d) { return "If used, this measure can only be broken down by one of the following groups of dimensions at a time:"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b870743803 = messageFormatterFn((function(  ) {
  return function(d) { return "If used, this measure will be aggregated to the level of detail of related dimensions:"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1259538438 = messageFormatterFn((function(  ) {
  return function(d) { return "If used, this dimension will stitch together the following fields:"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a903842824 = messageFormatterFn((function(  ) {
  return function(d) { return "If used, this measure will be ambiguously related to and so won't be broken down by the following:"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1760490227 = messageFormatterFn((function(  ) {
  return function(d) { return "This measure is ambiguously related to and so can't be broken down by the following:"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1862701528 = messageFormatterFn((function(  ) {
  return function(d) { return "If used, this dimension will be ambiguously related to and won't break down the following:"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1575618365 = messageFormatterFn((function(  ) {
  return function(d) { return "This dimension is ambiguously related to and so can't break down the following:"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2140642454 = messageFormatterFn((function(  ) {
  return function(d) { return "If used, this measure will be ambiguously related to and therefore ignored by the following filter:"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a639713562 = messageFormatterFn((function(  ) {
  return function(d) { return "If used, this measure will be ambiguously related to and therefore ignored by the following filters:"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1044667605 = messageFormatterFn((function(  ) {
  return function(d) { return "This measure is ambiguously related to and so is ignored by the filter:"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a326823503 = messageFormatterFn((function(  ) {
  return function(d) { return "This measure is ambiguously related to and so is ignored by these filters:"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1659900459 = messageFormatterFn((function(  ) {
  return function(d) { return "This measure can only be broken down by one of these dimensions at a time:"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1606146273 = messageFormatterFn((function(  ) {
  return function(d) { return "This measure can only be broken down by one of the following groups of dimensions at a time:"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b465223942 = messageFormatterFn((function(  ) {
  return function(d) { return "This measure is aggregated to the level of detail of related dimensions:"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a468837 = messageFormatterFn((function(  ) {
  return function(d) { return "This dimension stitches together the following fields:"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a525765021 = messageFormatterFn((function(  ) {
  return function(d) { return "This filter doesn't apply to unrelated fields:"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a165748941 = messageFormatterFn((function(  ) {
  return function(d) { return "This filter ignores the following measure because it is ambiguously related to it:"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b809907369 = messageFormatterFn((function(  ) {
  return function(d) { return "This filter ignores the following measures because it is ambiguously related to them:"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b936449498 = messageFormatterFn((function(  ) {
  return function(d) { return "This filter ignores the following field because it is not yet related to it:"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1325748051 = messageFormatterFn((function(  ) {
  return function(d) { return "This filter ignores the following fields because it is not yet related to them:"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1235078674 = messageFormatterFn((function(  ) {
  return function(d) { return "This field is not yet related to and so is ignored by these filters:"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b874671615 = messageFormatterFn((function(  ) {
  return function(d) { return "This field is not yet related to and so is ignored by the filter:"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b713233580 = messageFormatterFn((function(  ) {
  return function(d) { return "If used, this field won't yet be related to and so will be ignored by unrelated filters:"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b972259765 = messageFormatterFn((function(  ) {
  return function(d) { return "This measure is incompatible with one or more of the dimensions in this view."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a118283891 = messageFormatterFn((function(  ) {
  return function(d) { return "Table Calculation"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a243993184 = messageFormatterFn((function(  ) {
  return function(d) { return d.text + "…"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2109719842 = messageFormatterFn((function( plural, pluralFuncs ) {
  return function(d) { return plural(d.worksheetCount, 0, pluralFuncs.en, { one: function() { return "Also applies to worksheet: ";}, other: function() { return "Also applies to worksheets: ";} }); }
})(messageFormat.plural, {"en": Globalize("en").pluralGenerator()}), Globalize("en").pluralGenerator({}));
Globalize.b920809065 = messageFormatterFn((function(  ) {
  return function(d) { return "Comment"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a373822148 = messageFormatterFn((function(  ) {
  return function(d) { return ":"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a652087801 = messageFormatterFn((function(  ) {
  return function(d) { return "Add " + d.device + " Layout"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1853847311 = messageFormatterFn((function(  ) {
  return function(d) { return "Device Preview"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a334276526 = messageFormatterFn((function(  ) {
  return function(d) { return "Device Type"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b610940159 = messageFormatterFn((function(  ) {
  return function(d) { return "Model"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b591160366 = messageFormatterFn((function(  ) {
  return function(d) { return "Switch to Landscape Mode"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1138657842 = messageFormatterFn((function(  ) {
  return function(d) { return "Switch to Portrait Mode"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1693876027 = messageFormatterFn((function(  ) {
  return function(d) { return "Tableau Mobile App"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b765989666 = messageFormatterFn((function(  ) {
  return function(d) { return "Adjusts the preview frame to show available screen size for devices running Tableau Mobile"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1173620848 = messageFormatterFn((function(  ) {
  return function(d) { return "https://help.tableau.com/current/pro/desktop/en-us/dates_custom_date_formats.htm"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b719493425 = messageFormatterFn((function(  ) {
  return function(d) { return "Format Type"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b183591179 = messageFormatterFn((function(  ) {
  return function(d) { return "Format"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a590898897 = messageFormatterFn((function(  ) {
  return function(d) { return "Learn more"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a733957981 = messageFormatterFn((function(  ) {
  return function(d) { return "Automatic"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b168589764 = messageFormatterFn((function(  ) {
  return function(d) { return "Axis Editing Properties"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a116826502 = messageFormatterFn((function(  ) {
  return function(d) { return "Axis Range Options"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1670472719 = messageFormatterFn((function(  ) {
  return function(d) { return "Synchronize Axes"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1596155136 = messageFormatterFn((function(  ) {
  return function(d) { return "Axis Titles"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a434998165 = messageFormatterFn((function(  ) {
  return function(d) { return "Custom"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1231669957 = messageFormatterFn((function(  ) {
  return function(d) { return "Days"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1607536880 = messageFormatterFn((function(  ) {
  return function(d) { return "Fixed"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1278768609 = messageFormatterFn((function(  ) {
  return function(d) { return "Fixed end"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a247547112 = messageFormatterFn((function(  ) {
  return function(d) { return "Fixed start"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b970394972 = messageFormatterFn((function(  ) {
  return function(d) { return "General"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a125144769 = messageFormatterFn((function(  ) {
  return function(d) { return "Hours"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1581978018 = messageFormatterFn((function(  ) {
  return function(d) { return "Include zero"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1260422356 = messageFormatterFn((function(  ) {
  return function(d) { return "Independent"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1511552441 = messageFormatterFn((function(  ) {
  return function(d) { return "Independent axis ranges for each row or column"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a370610397 = messageFormatterFn((function(  ) {
  return function(d) { return "Insert"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1420855721 = messageFormatterFn((function(  ) {
  return function(d) { return "Interval"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a517883563 = messageFormatterFn((function(  ) {
  return function(d) { return "Enter a valid number."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2101016678 = messageFormatterFn((function(  ) {
  return function(d) { return "Enter a valid date."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b849525218 = messageFormatterFn((function(  ) {
  return function(d) { return "ISO Quarters"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1418024994 = messageFormatterFn((function(  ) {
  return function(d) { return "ISO Weekdays"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1304678114 = messageFormatterFn((function(  ) {
  return function(d) { return "ISO Weeks"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a466843701 = messageFormatterFn((function(  ) {
  return function(d) { return "ISO Years"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2046264182 = messageFormatterFn((function(  ) {
  return function(d) { return "Positive"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2108222031 = messageFormatterFn((function(  ) {
  return function(d) { return "Logarithmic"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1973292878 = messageFormatterFn((function(  ) {
  return function(d) { return "Logarithm base must be greater than one."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1209983741 = messageFormatterFn((function(  ) {
  return function(d) { return "Log axis origin must be greater than zero."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1229928788 = messageFormatterFn((function(  ) {
  return function(d) { return "Major Tick Marks"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1770199800 = messageFormatterFn((function(  ) {
  return function(d) { return "Major tick interval must be greater than zero."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1616793128 = messageFormatterFn((function(  ) {
  return function(d) { return "Minor Tick Marks"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a732038404 = messageFormatterFn((function(  ) {
  return function(d) { return "Minor tick interval must be greater than zero."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a157011665 = messageFormatterFn((function(  ) {
  return function(d) { return "Minutes"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b490170985 = messageFormatterFn((function(  ) {
  return function(d) { return "Months"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b932790308 = messageFormatterFn((function(  ) {
  return function(d) { return "None"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a22527915 = messageFormatterFn((function(  ) {
  return function(d) { return "Quarters"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b403943712 = messageFormatterFn((function(  ) {
  return function(d) { return "Range"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a121722113 = messageFormatterFn((function(  ) {
  return function(d) { return "Reset"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2072550554 = messageFormatterFn((function(  ) {
  return function(d) { return "Reversed"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1761425869 = messageFormatterFn((function(  ) {
  return function(d) { return "Scale"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1829038991 = messageFormatterFn((function(  ) {
  return function(d) { return "Seconds"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a642519329 = messageFormatterFn((function(  ) {
  return function(d) { return "End Extent"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1034593944 = messageFormatterFn((function(  ) {
  return function(d) { return "Start Extent"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1957939643 = messageFormatterFn((function(  ) {
  return function(d) { return "End Value"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a363942786 = messageFormatterFn((function(  ) {
  return function(d) { return "Start Value"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2140984811 = messageFormatterFn((function(  ) {
  return function(d) { return "Axis Extents"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a878507177 = messageFormatterFn((function(  ) {
  return function(d) { return "Show times"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1448908740 = messageFormatterFn((function(  ) {
  return function(d) { return "Subtitle"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1596485665 = messageFormatterFn((function(  ) {
  return function(d) { return "Symmetric"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2012938208 = messageFormatterFn((function(  ) {
  return function(d) { return "Synchronize dual axes"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1494018867 = messageFormatterFn((function(  ) {
  return function(d) { return "Tick interval"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a739891352 = messageFormatterFn((function(  ) {
  return function(d) { return "Tick interval (powers of)"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1431600146 = messageFormatterFn((function(  ) {
  return function(d) { return "Tick Marks"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b298200556 = messageFormatterFn((function(  ) {
  return function(d) { return "Tick origin"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2012363722 = messageFormatterFn((function(  ) {
  return function(d) { return "Title"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a495382896 = messageFormatterFn((function(  ) {
  return function(d) { return "Uniform"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1525100723 = messageFormatterFn((function(  ) {
  return function(d) { return "Uniform axis range for all rows or columns"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b733444312 = messageFormatterFn((function(  ) {
  return function(d) { return "Unit"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1614976974 = messageFormatterFn((function(  ) {
  return function(d) { return "Use Value"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a251521713 = messageFormatterFn((function(  ) {
  return function(d) { return "Weeks"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2023043528 = messageFormatterFn((function(  ) {
  return function(d) { return "Years"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1924410198 = messageFormatterFn((function(  ) {
  return function(d) { return "Top-left cell: "; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a392900639 = messageFormatterFn((function(  ) {
  return function(d) { return "Bottom-right cell: "; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b632638037 = messageFormatterFn((function(  ) {
  return function(d) { return ":"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b448990971 = messageFormatterFn((function(  ) {
  return function(d) { return "Apply"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1580369159 = messageFormatterFn((function(  ) {
  return function(d) { return "OK"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1069176069 = messageFormatterFn((function(  ) {
  return function(d) { return "Show template project contributors!"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a460159626 = messageFormatterFn((function(  ) {
  return function(d) { return "Hide template project contributors!"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1249357290 = messageFormatterFn((function(  ) {
  return function(d) { return "Edit Alias"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b971944610 = messageFormatterFn((function(  ) {
  return function(d) { return "OK"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1433062780 = messageFormatterFn((function(  ) {
  return function(d) { return "Cancel"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b448388890 = messageFormatterFn((function(  ) {
  return function(d) { return "Name"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1768604994 = messageFormatterFn((function(  ) {
  return function(d) { return "Show template project contributors!"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1917345859 = messageFormatterFn((function(  ) {
  return function(d) { return "Hide template project contributors!"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2024236342 = messageFormatterFn((function(  ) {
  return function(d) { return "Set Height"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1554774750 = messageFormatterFn((function(  ) {
  return function(d) { return "Height in pixels"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a49045741 = messageFormatterFn((function(  ) {
  return function(d) { return "Set Width"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1939523483 = messageFormatterFn((function(  ) {
  return function(d) { return "Width in pixels"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b348759058 = messageFormatterFn((function(  ) {
  return function(d) { return "Reset"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1148749899 = messageFormatterFn((function(  ) {
  return function(d) { return "Cancel"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1476002793 = messageFormatterFn((function(  ) {
  return function(d) { return "OK"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a326072280 = messageFormatterFn((function(  ) {
  return function(d) { return "EPS URL cannot be empty"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a463990052 = messageFormatterFn((function(  ) {
  return function(d) { return "OK"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a760485858 = messageFormatterFn((function(  ) {
  return function(d) { return "Cancel"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a905432854 = messageFormatterFn((function(  ) {
  return function(d) { return d.fieldName + " can't be used in explanations because the required aggregation (AVG) can't be used with this measure."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1168501500 = messageFormatterFn((function(  ) {
  return function(d) { return d.fieldName + " can't be used in explanations because it's being automatically hidden."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a923280706 = messageFormatterFn((function(  ) {
  return function(d) { return d.fieldName + " can't be used in explanations because it isn't a discrete dimension."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b499768624 = messageFormatterFn((function(  ) {
  return function(d) { return d.fieldName + " can't be used in explanations because it isn't a continuous measure."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a501847795 = messageFormatterFn((function(  ) {
  return function(d) { return d.fieldName + " can't be used in explanations because its domain information isn't available."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a937701117 = messageFormatterFn((function(  ) {
  return function(d) { return d.fieldName + " can't be used in explanations because geometry, latitude and longitude aren't supported."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a583980919 = messageFormatterFn((function(  ) {
  return function(d) { return d.fieldName + " can't be used in explanations because it has more than 500 members."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a657312802 = messageFormatterFn((function(  ) {
  return function(d) { return "Fields that aren't valid can't be selected for consideration."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a136238613 = messageFormatterFn((function(  ) {
  return function(d) { return d.fieldName + " can't be used in explanations because its field type isn't supported. Supported field types for dimensions: calculated fields and physical columns defined in the database for the data source."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1856451315 = messageFormatterFn((function(  ) {
  return function(d) { return d.fieldName + " can't be used in explanations because its field type isn't supported. Supported field types for measures: calculated fields and physical columns defined in the database for the data source."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a570251009 = messageFormatterFn((function(  ) {
  return function(d) { return d.fieldName + " can't be used in explanations because its data type isn't supported. Supported data types for dimensions: string, number and boolean."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b327509767 = messageFormatterFn((function(  ) {
  return function(d) { return d.fieldName + " can't be used in explanations because its data type isn't supported. Supported data type for measures: number."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1216478623 = messageFormatterFn((function(  ) {
  return function(d) { return "Data source"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b331588855 = messageFormatterFn((function(  ) {
  return function(d) { return "Not connected"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2029321407 = messageFormatterFn((function(  ) {
  return function(d) { return "Unsupported connection type"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a110145616 = messageFormatterFn((function(  ) {
  return function(d) { return "None of the data sources in this workbook can be used in explanations."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b45329311 = messageFormatterFn((function(  ) {
  return function(d) { return "Fields"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a533666082 = messageFormatterFn((function(  ) {
  return function(d) { return "Explanations"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1720406386 = messageFormatterFn((function(  ) {
  return function(d) { return "Explain the Average"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a303258684 = messageFormatterFn((function(  ) {
  return function(d) { return "Average Value of the Mark"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a631106785 = messageFormatterFn((function(  ) {
  return function(d) { return "Number of Records"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b621613082 = messageFormatterFn((function(  ) {
  return function(d) { return "Extreme Values"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1589041202 = messageFormatterFn((function(  ) {
  return function(d) { return "Contributing Dimensions"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1704673350 = messageFormatterFn((function(  ) {
  return function(d) { return "Contributing Measures"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1093276776 = messageFormatterFn((function(  ) {
  return function(d) { return "Null Values"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1477325359 = messageFormatterFn((function(  ) {
  return function(d) { return "Contributing Single Values"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b8468268 = messageFormatterFn((function(  ) {
  return function(d) { return "This explanation type indicates if one or more records have values that are significantly higher or lower than most records in the target measure."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1070272938 = messageFormatterFn((function(  ) {
  return function(d) { return "Contributing to the value of a measure:\n\nThis explanation type indicates when the distribution of values across an unvisualized dimension may be contributing to the aggregate value of the analyzed mark. This type of explanation is used for sums, counts, and averages.\n\nUnique Attributes:\n\nThis explanation type can also indicate that the distribution of values across an unvisualized dimension in the analyzed mark is unusual in contrast to the distribution of values for all other marks in the view.\n\nAn unvisualized dimension is a dimension that exists in the data source, but isn't currently being used in the view."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1072818304 = messageFormatterFn((function(  ) {
  return function(d) { return "This explanation type describes when the count of the underlying records is correlated to the SUM of the target measure."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1339510085 = messageFormatterFn((function(  ) {
  return function(d) { return "This explanation type describes when the average of the target measure is correlated to the SUM."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b890165454 = messageFormatterFn((function(  ) {
  return function(d) { return "This explanation type indicates the average of an unvisualized measure may be contributing to the aggregate value of the analyzed mark.\n\nAn unvisualized measure is a measure that exists in the data source, but isn't currently being used in the view."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1363098009 = messageFormatterFn((function(  ) {
  return function(d) { return "Contributing to the value of a measure:\n\nThis explanation type indicates when a single value in an unvisualized dimension may be contributing to the aggregate value of the analyzed mark. This explanation reveals when every underlying record of a dimension has the same value.\n\nUnique Attributes:\n\nThis explanation type can also indicate that the value of the analyzed mark is unusual in contrast to the distribution of values for all other marks in the view.\n\nAn unvisualized dimension is a dimension that exists in the data source, but isn't currently being used in the view."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1019045993 = messageFormatterFn((function(  ) {
  return function(d) { return "This explanation type reveals how the fraction of target measure values that are null might be contributing to the aggregated value of that measure."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1420335693 = messageFormatterFn((function(  ) {
  return function(d) { return "Explain the Average."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a709938429 = messageFormatterFn((function(  ) {
  return function(d) { return "Reset"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1101502162 = messageFormatterFn((function(  ) {
  return function(d) { return "Include None"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1694387258 = messageFormatterFn((function(  ) {
  return function(d) { return "Explain Data Settings"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a809196999 = messageFormatterFn((function(  ) {
  return function(d) { return "Always Include"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a207280036 = messageFormatterFn((function(  ) {
  return function(d) { return "Never Include"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b904238759 = messageFormatterFn((function(  ) {
  return function(d) { return "Automatic"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1746702295 = messageFormatterFn((function(  ) {
  return function(d) { return "Automatic (Exclude when number of members is large)"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a245632747 = messageFormatterFn((function(  ) {
  return function(d) { return "Cannot be used"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b594338285 = messageFormatterFn((function(  ) {
  return function(d) { return "Choose the fields used for explaining data:"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b933081127 = messageFormatterFn((function(  ) {
  return function(d) { return "Explanation types to show in this workbook:"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a879314791 = messageFormatterFn((function(  ) {
  return function(d) { return "Learn more"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b788227341 = messageFormatterFn((function(  ) {
  return function(d) { return "Name"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a126787926 = messageFormatterFn((function(  ) {
  return function(d) { return "Include"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1759472234 = messageFormatterFn((function(  ) {
  return function(d) { return "Unable to get field metadata from the datasource."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a576099872 = messageFormatterFn((function(  ) {
  return function(d) { return "Mark Details"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1698141423 = messageFormatterFn((function(  ) {
  return function(d) { return "Choose fields for '" + d.datasourceCaption + "'"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2048269026 = messageFormatterFn((function(  ) {
  return function(d) { return "Analyzed at " + d.datetime; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b498356647 = messageFormatterFn((function(  ) {
  return function(d) { return "Help"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1433963486 = messageFormatterFn((function(  ) {
  return function(d) { return "Edit"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b925553520 = messageFormatterFn((function(  ) {
  return function(d) { return "Open explanation"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1132460664 = messageFormatterFn((function(  ) {
  return function(d) { return "Hide explanation"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a883043162 = messageFormatterFn((function(  ) {
  return function(d) { return "Hide"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1752884153 = messageFormatterFn((function(  ) {
  return function(d) { return "Go back"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b849342261 = messageFormatterFn((function(  ) {
  return function(d) { return "Analyzed mark"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1373066907 = messageFormatterFn((function(  ) {
  return function(d) { return "Explore underlying values for " + d.analyzedMark; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2108608156 = messageFormatterFn((function(  ) {
  return function(d) { return "For the measure value listed below, why is " + d.analyzedMark + " different from other marks?"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1097237003 = messageFormatterFn((function(  ) {
  return function(d) { return "For the measure values listed below, why is " + d.analyzedMark + " different from other marks?"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1752657781 = messageFormatterFn((function(  ) {
  return function(d) { return "This explanation can't be generated."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b344060757 = messageFormatterFn((function(  ) {
  return function(d) { return "What has changed unexpectedly for " + d.analyzedMark + "?"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1817733801 = messageFormatterFn((function(  ) {
  return function(d) { return "No unexpected data changes found."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b272939593 = messageFormatterFn((function(  ) {
  return function(d) { return "Data Change Radar is not enabled for this workbook."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a312990738 = messageFormatterFn((function(  ) {
  return function(d) { return "Other things to explore"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a125050597 = messageFormatterFn((function(  ) {
  return function(d) { return "Other than measure values, why is " + d.analyzedMark + " different from other marks?"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1553953151 = messageFormatterFn((function(  ) {
  return function(d) { return d.numFields + " fields were skipped to reduce time and resources required for analysis. Click Explain All to run a longer analysis that considers these additional fields."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1100016765 = messageFormatterFn((function(  ) {
  return function(d) { return "Explain All"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2086495275 = messageFormatterFn((function(  ) {
  return function(d) { return "What is unique about " + d.analyzedMark + "?"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1996542876 = messageFormatterFn((function(  ) {
  return function(d) { return "Unique Attributes"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b905236952 = messageFormatterFn((function(  ) {
  return function(d) { return "Average values"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a481730654 = messageFormatterFn((function(  ) {
  return function(d) { return "Within expected range compared to other marks"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1390650712 = messageFormatterFn((function(  ) {
  return function(d) { return "High"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a374668388 = messageFormatterFn((function(  ) {
  return function(d) { return "Low"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1195699525 = messageFormatterFn((function(  ) {
  return function(d) { return "Similar"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a616380674 = messageFormatterFn((function(  ) {
  return function(d) { return "sum of " + d.measureCaption; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1818336144 = messageFormatterFn((function(  ) {
  return function(d) { return "average of " + d.measureCaption; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b843338498 = messageFormatterFn((function(  ) {
  return function(d) { return "count of " + d.measureCaption; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a225011714 = messageFormatterFn((function(  ) {
  return function(d) { return "distinct count of " + d.measureCaption; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1830940435 = messageFormatterFn((function(  ) {
  return function(d) { return "median of " + d.measureCaption; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1598063475 = messageFormatterFn((function(  ) {
  return function(d) { return "maximum of " + d.measureCaption; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1448853983 = messageFormatterFn((function(  ) {
  return function(d) { return "minimum of " + d.measureCaption; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1037322281 = messageFormatterFn((function(  ) {
  return function(d) { return "standard deviation of " + d.measureCaption; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1680257612 = messageFormatterFn((function(  ) {
  return function(d) { return "standard deviation (from population) of " + d.measureCaption; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a995121126 = messageFormatterFn((function(  ) {
  return function(d) { return "variance of " + d.measureCaption; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1648591017 = messageFormatterFn((function(  ) {
  return function(d) { return "variance (from population) of " + d.measureCaption; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1698176718 = messageFormatterFn((function(  ) {
  return function(d) { return d.N + "st/nd/rd/th percentile of " + d.measureCaption; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b977962382 = messageFormatterFn((function(  ) {
  return function(d) { return "Expected Value"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1693936404 = messageFormatterFn((function(  ) {
  return function(d) { return "The mark value is within the expected range.\n\nThe expected range is from the 15th to the 85th percentile that a statistical model predicts for the analyzed mark based on visualized dimensions only."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2031915484 = messageFormatterFn((function(  ) {
  return function(d) { return "The aggregated value of the analyzed mark is higher than predicted, but within the expected range."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a207257698 = messageFormatterFn((function(  ) {
  return function(d) { return "The aggregated value of the analyzed mark is lower than predicted, but within the expected range."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1853583188 = messageFormatterFn((function(  ) {
  return function(d) { return "Why is this the median value?"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b639571323 = messageFormatterFn((function(  ) {
  return function(d) { return "Why is this slightly high compared to other marks?"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b579111967 = messageFormatterFn((function(  ) {
  return function(d) { return "Why is this slightly low compared to other marks?"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a970990223 = messageFormatterFn((function(  ) {
  return function(d) { return "Why is this high compared to other marks?"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b804253033 = messageFormatterFn((function(  ) {
  return function(d) { return "Why is this low compared to other marks?"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1698695190 = messageFormatterFn((function(  ) {
  return function(d) { return "Expected Range"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b942567760 = messageFormatterFn((function(  ) {
  return function(d) { return d.aggTargetMeasureName + " for " + d.analyzedMark + " is expected to be between " + d.low + " and " + d.high + ", given the model of current values in the viz. This takes into account the filters set on " + d.fieldCaptions + "."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b383720033 = messageFormatterFn((function(  ) {
  return function(d) { return d.aggTargetMeasureName + " for " + d.analyzedMark + " is expected to be between " + d.low + " and " + d.high + ", given the model of current values in the viz."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1632603357 = messageFormatterFn((function(  ) {
  return function(d) { return d.aggTargetMeasureName + " for " + d.analyzedMark + " is expected to be between " + d.low + " and " + d.high + ", given the model of current values in the viz. This takes into account the filter set on " + d.fieldCaption + "."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a129391110 = messageFormatterFn((function(  ) {
  return function(d) { return "Unable to complete this operation"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1774650709 = messageFormatterFn((function(  ) {
  return function(d) { return "Explain Data can't generate this view."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1122502818 = messageFormatterFn((function(  ) {
  return function(d) { return "Tableau couldn't complete explanation analysis because the selected mark changed. Close the Explain Data pane and try again."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1270315769 = messageFormatterFn((function(  ) {
  return function(d) { return "Tableau couldn't complete explanation analysis because the visualization changed. Close the Explain Data pane and try again."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1284667773 = messageFormatterFn((function(  ) {
  return function(d) { return "An error occured in showing explanations."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1143668031 = messageFormatterFn((function(  ) {
  return function(d) { return "One record out of " + d.numberOfRows + " has an extreme value, which is making " + d.aggTargetMeasureName + " higher for the analyzed mark " + d.analyzedMark + "."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b456825819 = messageFormatterFn((function(  ) {
  return function(d) { return "One record out of " + d.numberOfRows + " has an extreme value, which is making " + d.aggTargetMeasureName + " lower for the analyzed mark " + d.analyzedMark + "."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1543972041 = messageFormatterFn((function(  ) {
  return function(d) { return d.X + " records out of " + d.numberOfRows + " have extreme values, which are making " + d.aggTargetMeasureName + " higher for the analyzed mark " + d.analyzedMark + "."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a876798237 = messageFormatterFn((function(  ) {
  return function(d) { return d.X + " records out of " + d.numberOfRows + " have extreme values, which are making " + d.aggTargetMeasureName + " lower for the analyzed mark " + d.analyzedMark + "."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1930588643 = messageFormatterFn((function(  ) {
  return function(d) { return d.numberOfValues + " extreme values"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1177948526 = messageFormatterFn((function(  ) {
  return function(d) { return "One extreme value: " + d.outlierValue; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1383665725 = messageFormatterFn((function(  ) {
  return function(d) { return "Distribution of " + d.measureCaption + " in the analyzed mark"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a176521379 = messageFormatterFn((function(  ) {
  return function(d) { return "This chart shows all records of " + d.measureCaption + " for " + d.analyzedMark + ". The extreme value " + d.outlierValue + " is highlighted."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b638053221 = messageFormatterFn((function(  ) {
  return function(d) { return "This chart shows all records of " + d.measureCaption + " for " + d.analyzedMark + ". The extreme values are highlighted."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a832664141 = messageFormatterFn((function(  ) {
  return function(d) { return "Visualize the Difference"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a924438283 = messageFormatterFn((function(  ) {
  return function(d) { return "When the extreme values are excluded, the " + d.aggTargetMeasureName + " changes from " + d.measureValue + " to " + d.measureValueWithoutOutlier + "."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b608493565 = messageFormatterFn((function(  ) {
  return function(d) { return "When the extreme value is excluded, the " + d.aggTargetMeasureName + " changes from " + d.measureValue + " to " + d.measureValueWithoutOutlier + "."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b242654854 = messageFormatterFn((function(  ) {
  return function(d) { return "On average, record values in " + d.analyzedMark + " are higher than record values for other marks. This may explain why " + d.aggTargetMeasureName + " is high."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1511901772 = messageFormatterFn((function(  ) {
  return function(d) { return "On average, record values in " + d.analyzedMark + " are lower than record values for other marks. This may explain why " + d.aggTargetMeasureName + " is low."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2132883848 = messageFormatterFn((function(  ) {
  return function(d) { return "The average of record values in " + d.analyzedMark + " may be affecting " + d.aggTargetMeasureName + "."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a733779684 = messageFormatterFn((function(  ) {
  return function(d) { return "Average " + d.baseColumnCaption + ": " + d.averageOfRecords; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a507049619 = messageFormatterFn((function(  ) {
  return function(d) { return "Annual Trend"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b309089340 = messageFormatterFn((function(  ) {
  return function(d) { return "Seasonal Variations"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a47088092 = messageFormatterFn((function(  ) {
  return function(d) { return "Monthly Variations"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1012248281 = messageFormatterFn((function(  ) {
  return function(d) { return "The trending of annual average may explain why  " + d.aggTargetMeasureName + " of " + d.analyzedMark + " is higher"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1283870465 = messageFormatterFn((function(  ) {
  return function(d) { return "The trending of annual average may explain why " + d.aggTargetMeasureName + " of " + d.analyzedMark + " is lower"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1342629033 = messageFormatterFn((function(  ) {
  return function(d) { return "The trending of annual average may affect " + d.aggTargetMeasureName + " of " + d.analyzedMark; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1196548418 = messageFormatterFn((function(  ) {
  return function(d) { return "The annual cycles may explain why " + d.aggTargetMeasureName + " of " + d.analyzedMark + " is higher"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1619678344 = messageFormatterFn((function(  ) {
  return function(d) { return "The annual cycles may explain why " + d.aggTargetMeasureName + " of " + d.analyzedMark + " is lower"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b75707918 = messageFormatterFn((function(  ) {
  return function(d) { return "The annual cycles may affect " + d.aggTargetMeasureName + " of " + d.analyzedMark; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b612806155 = messageFormatterFn((function(  ) {
  return function(d) { return "Average of " + d.baseColumnCaption + " by Year"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1268817546 = messageFormatterFn((function(  ) {
  return function(d) { return "Average of " + d.baseColumnCaption + " by Quarter"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a716158494 = messageFormatterFn((function(  ) {
  return function(d) { return "Average of " + d.baseColumnCaption + " by Month"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1257407761 = messageFormatterFn((function(  ) {
  return function(d) { return "Average of " + d.baseColumnCaption; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a267509768 = messageFormatterFn((function(  ) {
  return function(d) { return "No Explanation"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1026853118 = messageFormatterFn((function(  ) {
  return function(d) { return "When the analyzed mark value is outside of the expected range and no statistical models including unvisualized predictors can explain the mark, no explanations are generated.\n\nThe expected range is from the 15th to the 85th percentile that a statistical model predicts for the analyzed mark based on visualized dimensions only."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1667797660 = messageFormatterFn((function(  ) {
  return function(d) { return "The fields " + d.consideredFields + " were each considered as predictors but none adequately explain the mark."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1901110312 = messageFormatterFn((function(  ) {
  return function(d) { return d.numberOfFields + " fields were each considered as predictors, such as " + d.fieldCaption1 + ", " + d.fieldCaption2 + " and " + d.fieldCaption3 + ", but none adequately explain the mark."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a568685217 = messageFormatterFn((function(  ) {
  return function(d) { return "Additional fields were not available for consideration in this analysis."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b636646750 = messageFormatterFn((function(  ) {
  return function(d) { return "The field " + d.fieldCaption + " was considered as a predictor, but did not adequately explain the mark."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1960692410 = messageFormatterFn((function(  ) {
  return function(d) { return "Random Variation"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a12113600 = messageFormatterFn((function(  ) {
  return function(d) { return "When the analyzed mark has a low number of records, there may not be enough data available for Explain Data to form a statistically significant explanation. If the mark’s value is outside the expected range, Explain Data can’t determine whether this unexpected value is being caused by random variation or by a meaningful difference in the underlying records.\n\nThe expected range is from the 15th to the 85th percentile that a statistical model predicts for the analyzed mark based on visualized dimensions only."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1183670793 = messageFormatterFn((function(  ) {
  return function(d) { return "The analyzed mark " + d.analyzedMark + " is made of a single record."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b305760175 = messageFormatterFn((function(  ) {
  return function(d) { return "The analyzed mark " + d.analyzedMark + " is made of " + d.numRows + " records."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a884344614 = messageFormatterFn((function(  ) {
  return function(d) { return "The analyzed mark " + d.analyzedMark + " is made of a single record. Its unexpected value is likely to be caused by random variation rather than a meaningful difference in the underlying records."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b195409362 = messageFormatterFn((function(  ) {
  return function(d) { return "The analyzed mark " + d.analyzedMark + " is made of " + d.numRows + " records. Its unexpected value is likely to be caused by random variation rather than a meaningful difference in the underlying records."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2129165286 = messageFormatterFn((function(  ) {
  return function(d) { return "Underlying Characteristics"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1437110176 = messageFormatterFn((function(  ) {
  return function(d) { return "Relevant Trends"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1694660454 = messageFormatterFn((function(  ) {
  return function(d) { return "Which time trends have the largest influence on " + d.aggTargetMeasureName + " for " + d.analyzedMark + "? Time trends are listed in order of explanatory power."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a214914400 = messageFormatterFn((function(  ) {
  return function(d) { return "These explanations show relationships between the underlying records that make up " + d.analyzedMark + " and the value of " + d.aggTargetMeasureName + ".\n\n**Extreme Values** Are one or multiple records skewing " + d.aggTargetMeasureName + "?\n\n**Null Values** Is the presence of null values influencing " + d.aggTargetMeasureName + "?\n\nFor SUM aggregations:\n\n**Number of Records** Is the sum of " + d.measureName + " high/low because there are many/few records?\n\n**Average Value of Mark** Is the sum of " + d.measureName + " high/low because individual records have high/low values?"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b774413285 = messageFormatterFn((function(  ) {
  return function(d) { return d.aggTargetMeasureName + " may be high because of the high number of records in " + d.analyzedMark + "."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1968935029 = messageFormatterFn((function(  ) {
  return function(d) { return d.aggTargetMeasureName + " may be low because of the low number of records in " + d.analyzedMark + "."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1347952953 = messageFormatterFn((function(  ) {
  return function(d) { return "The number of records in " + d.analyzedMark + " may be affecting " + d.aggTargetMeasureName + "."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a567205776 = messageFormatterFn((function(  ) {
  return function(d) { return "Number of Records"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a689562275 = messageFormatterFn((function(  ) {
  return function(d) { return "Number of Records: " + d.numberOfRows; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1948816291 = messageFormatterFn((function(  ) {
  return function(d) { return d.unvisualizedMeasureName + ": " + d.unvisualizedMeasureValue; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a928679381 = messageFormatterFn((function(  ) {
  return function(d) { return "Null Values: " + d.percentNull; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1526248199 = messageFormatterFn((function(  ) {
  return function(d) { return d.analyzedMark + " has more records with " + d.dimensionName + ": " + d.markValue + " than other marks, and " + d.baseColumnCaption + " tends to be higher for " + d.markValue + ". This may explain why " + d.aggTargetMeasureName + " is higher for " + d.analyzedMark + "."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b370699489 = messageFormatterFn((function(  ) {
  return function(d) { return d.analyzedMark + " has more records with " + d.dimensionName + ": " + d.markValue + " than other marks, and " + d.baseColumnCaption + " tends to be lower for " + d.markValue + ". This may explain why " + d.aggTargetMeasureName + " is lower for " + d.analyzedMark + "."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1568358711 = messageFormatterFn((function(  ) {
  return function(d) { return d.analyzedMark + " has more records with " + d.dimensionName + ": " + d.markValue + " than other marks, which may be affecting " + d.aggTargetMeasureName + " for " + d.analyzedMark + "."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a453842609 = messageFormatterFn((function(  ) {
  return function(d) { return d.analyzedMark + " has fewer records with " + d.dimensionName + ": " + d.markValue + " than other marks, and " + d.baseColumnCaption + " tends to be lower for " + d.markValue + ". This may explain why " + d.aggTargetMeasureName + " is higher for " + d.analyzedMark + "."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1790766539 = messageFormatterFn((function(  ) {
  return function(d) { return d.analyzedMark + " has fewer records with " + d.dimensionName + ": " + d.markValue + " than other marks, and " + d.baseColumnCaption + " tends to be higher for " + d.markValue + ". This may explain why " + d.aggTargetMeasureName + " is lower for " + d.analyzedMark + "."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b453193633 = messageFormatterFn((function(  ) {
  return function(d) { return d.analyzedMark + " has fewer records with " + d.dimensionName + ": " + d.markValue + " than other marks, which may be affecting " + d.aggTargetMeasureName + " for " + d.analyzedMark + "."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2067765525 = messageFormatterFn((function(  ) {
  return function(d) { return "Average of " + d.baseColumnCaption + " for " + d.dimension + ": " + d.primaryDisplayValue; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1030244196 = messageFormatterFn((function(  ) {
  return function(d) { return "Percent of records with " + d.dimension + ": " + d.primaryDisplayValue; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1524539770 = messageFormatterFn((function(  ) {
  return function(d) { return ", "; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a533729980 = messageFormatterFn((function(  ) {
  return function(d) { return ": "; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b36044217 = messageFormatterFn((function(  ) {
  return function(d) { return "This explanation type indicates when every underlying record of a dimension has the same value. It can also reveal when a single value in an unvisualized dimension has a substantial contribution to the mark’s value."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b285244567 = messageFormatterFn((function(  ) {
  return function(d) { return "All records in the analyzed mark have the same single value in the unvisualized dimension, which is unusual in contrast to the distribution of values for all other marks in the view."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b248065048 = messageFormatterFn((function(  ) {
  return function(d) { return "Contributing Single Value"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a719464198 = messageFormatterFn((function(  ) {
  return function(d) { return "Relevant Single Value"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1583864244 = messageFormatterFn((function(  ) {
  return function(d) { return "Every underlying record of the analyzed mark has the value of " + d.markValue + " for " + d.dimensionName + "."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a307766767 = messageFormatterFn((function(  ) {
  return function(d) { return "The distribution of " + d.dimensionName + " is increasing the expected " + d.aggTargetMeasureName + " in the analyzed mark. All records in the analyzed mark have " + d.dimensionName + ": " + d.primaryDisplayValue + "."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a282731319 = messageFormatterFn((function(  ) {
  return function(d) { return "The distribution of " + d.dimensionName + " is decreasing the expected " + d.aggTargetMeasureName + " in the analyzed mark. All records in the analyzed mark have " + d.dimensionName + ": " + d.primaryDisplayValue + "."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b686577439 = messageFormatterFn((function(  ) {
  return function(d) { return "The distribution of " + d.dimensionName + " may be contributing to the expected " + d.aggTargetMeasureName + ". All records in the analyzed mark have " + d.dimensionName + ": " + d.primaryDisplayValue + "."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1310646390 = messageFormatterFn((function(  ) {
  return function(d) { return "Shared explanations might change if the data changes."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1608021407 = messageFormatterFn((function(  ) {
  return function(d) { return "The shared explanation is no longer available."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a721057033 = messageFormatterFn((function(  ) {
  return function(d) { return d.consideredFieldsNum + " of " + d.totalFieldsNum + " fields analyzed"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1757018780 = messageFormatterFn((function(  ) {
  return function(d) { return "Included (" + d.fieldNum + ")"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1648397354 = messageFormatterFn((function(  ) {
  return function(d) { return "Excluded (" + d.fieldNum + ")"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b601011515 = messageFormatterFn((function(  ) {
  return function(d) { return "datasource: " + d.datasourceCaption; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b140452210 = messageFormatterFn((function(  ) {
  return function(d) { return "Analyzed Fields"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1246222372 = messageFormatterFn((function(  ) {
  return function(d) { return "When you run Explain Data on a mark, a statistical analysis is run on the aggregated mark, and then on possibly related fields from the data source that aren't represented in the current view. Explain Data does not include every column from the data source in the analysis. The fields considered in the analysis are listed below."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b336528797 = messageFormatterFn((function(  ) {
  return function(d) { return "Learn More"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1318228176 = messageFormatterFn((function(  ) {
  return function(d) { return "When you select a mark in your viz and run Explain Data, Tableau analyzes the data in the view and in your connected data source. Based on the mark you selected, this pane shows possible explanations for the value, while considering other record values in the data source. Click section names below to explore possible explanations."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a550148307 = messageFormatterFn((function(  ) {
  return function(d) { return "Which dimensions have the largest influence on " + d.aggTargetMeasureName + " for " + d.analyzedMark + "? Dimensions are listed in order of explanatory power."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1900207324 = messageFormatterFn((function(  ) {
  return function(d) { return "Contributing Dimensions"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1125684730 = messageFormatterFn((function(  ) {
  return function(d) { return "Other Dimensions of Interest"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1667888117 = messageFormatterFn((function(  ) {
  return function(d) { return "This explanation type offers suggestions about what makes " + d.analyzedMark + " unique compared to other marks, even though the dimensions listed are uncorrelated with measure values shown in the viz.\n\nFor example, this explanation type could surface information like, “This sales representative was the only one who owns a cat,” or “The employees in New York had different distributions of pets than employees in other states."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a453676638 = messageFormatterFn((function(  ) {
  return function(d) { return "The distribution of records for " + d.dimensionName + " is different for " + d.analyzedMark + " than other marks, but no correlations were found with the measure values shown in the viz."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b438761568 = messageFormatterFn((function(  ) {
  return function(d) { return "The distribution of records for " + d.dimensionName + " is correlated with " + d.aggTargetMeasureName + ", which may explain why " + d.aggTargetMeasureName + " is higher for " + d.analyzedMark + "."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1681012890 = messageFormatterFn((function(  ) {
  return function(d) { return "The distribution of records for " + d.dimensionName + " is correlated with " + d.aggTargetMeasureName + ", which may explain why " + d.aggTargetMeasureName + " is lower for " + d.analyzedMark + "."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1940847952 = messageFormatterFn((function(  ) {
  return function(d) { return "The distribution of " + d.dimensionName + " is correlated with " + d.aggTargetMeasureName + ", though the analyzed mark " + d.analyzedMark + " is still in the expected range."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1627673150 = messageFormatterFn((function(  ) {
  return function(d) { return "Distribution of records by " + d.dimension; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1360973833 = messageFormatterFn((function(  ) {
  return function(d) { return "Average of " + d.baseColumnCaption + " by " + d.dimension; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b675759611 = messageFormatterFn((function(  ) {
  return function(d) { return "Which measures have the largest influence on " + d.aggTargetMeasureName + " for " + d.analyzedMark + "? Measures are listed in order of explanatory power."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1080261323 = messageFormatterFn((function(  ) {
  return function(d) { return "Contributing Measures"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2023229133 = messageFormatterFn((function(  ) {
  return function(d) { return d.analyzedMark + " has a higher average " + d.measureName + " than other marks. This value is associated with higher values of " + d.aggTargetMeasureName + " and may explain why it is higher."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1463046503 = messageFormatterFn((function(  ) {
  return function(d) { return d.analyzedMark + " has a lower average " + d.measureName + " than other marks. This value is associated with lower values of " + d.aggTargetMeasureName + " and may explain why it is lower."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1473558598 = messageFormatterFn((function(  ) {
  return function(d) { return d.analyzedMark + " has a lower average " + d.measureName + " than other marks. This value is associated with higher values of " + d.aggTargetMeasureName + " and may explain why it is higher."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1819177362 = messageFormatterFn((function(  ) {
  return function(d) { return d.analyzedMark + " has a higher average " + d.measureName + " than other marks. This value is associated with lower values of " + d.aggTargetMeasureName + " and may explain why it is lower."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a953148355 = messageFormatterFn((function(  ) {
  return function(d) { return d.analyzedMark + " has an average " + d.measureName + " that may be influencing the values of " + d.aggTargetMeasureName + "."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1609416682 = messageFormatterFn((function(  ) {
  return function(d) { return "Average of " + d.measureName + " and " + d.targetMeasureName; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b162205877 = messageFormatterFn((function(  ) {
  return function(d) { return "Percent of Nulls in " + d.targetMeasureName; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b343557566 = messageFormatterFn((function(  ) {
  return function(d) { return d.aggTargetMeasureName + " may be high because of the percentage of null records in " + d.analyzedMark + "."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b431015804 = messageFormatterFn((function(  ) {
  return function(d) { return d.aggTargetMeasureName + " may be low because of the percentage of null records in " + d.analyzedMark + "."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a597204718 = messageFormatterFn((function(  ) {
  return function(d) { return "The percent of null records in " + d.analyzedMark + " may be affecting " + d.aggTargetMeasureName + "."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1523851603 = messageFormatterFn((function(  ) {
  return function(d) { return "Top Contributors"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a194041485 = messageFormatterFn((function(  ) {
  return function(d) { return "Which dimensions have the most influence on " + d.aggTargetMeasureName + " for " + d.analyzedMark + "?"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1716029910 = messageFormatterFn((function(  ) {
  return function(d) { return "These values for " + d.dimensionName + " were the top contributors to " + d.aggTargetMeasureName + " for " + d.analyzedMark + "."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a97938105 = messageFormatterFn((function(  ) {
  return function(d) { return "This value for " + d.dimensionName + " was the top contributor to " + d.aggTargetMeasureName + " for " + d.analyzedMark + "."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1025137522 = messageFormatterFn((function(  ) {
  return function(d) { return "Top contributing " + d.dimensionName; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a527221595 = messageFormatterFn((function(  ) {
  return function(d) { return "Can't analyze measures that use this aggregation. Try using an aggregation type of Sum, Average, Count, or Count Distinct."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1021111733 = messageFormatterFn((function(  ) {
  return function(d) { return "Can't analyze measures that use table calculations."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1152134874 = messageFormatterFn((function(  ) {
  return function(d) { return "Can't analyze the aggregation of this measure. Try changing the aggregation to Sum or Count."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a313055568 = messageFormatterFn((function(  ) {
  return function(d) { return "Can't analyze measures that are null values."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1960767400 = messageFormatterFn((function(  ) {
  return function(d) { return "Can't analyze measures that are special values."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1249258283 = messageFormatterFn((function(  ) {
  return function(d) { return "Can't analyze measures that use forecasting."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1470448987 = messageFormatterFn((function(  ) {
  return function(d) { return "Can't analyze this measure because too few marks have values for this measure."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b937710124 = messageFormatterFn((function(  ) {
  return function(d) { return "This mark can't be analyzed for explanations."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a579728369 = messageFormatterFn((function(  ) {
  return function(d) { return "This explanation can't be displayed in a new worksheet because the data has changed. To refresh explanations, run Explain Data again."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b841049085 = messageFormatterFn((function(  ) {
  return function(d) { return "Share this explanation"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b581971886 = messageFormatterFn((function(  ) {
  return function(d) { return "Share explanation"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b31362834 = messageFormatterFn((function(  ) {
  return function(d) { return "You can't share explanations when the original mark is not selected or available. Reselect the mark to share its related explanations."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b426729887 = messageFormatterFn((function(  ) {
  return function(d) { return "% of Number of Records"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a272951469 = messageFormatterFn((function(  ) {
  return function(d) { return "Record Details"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2011364762 = messageFormatterFn((function(  ) {
  return function(d) { return "Include or Exclude Extreme Values"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1906714412 = messageFormatterFn((function(  ) {
  return function(d) { return "Include extreme values"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b324775003 = messageFormatterFn((function(  ) {
  return function(d) { return "Explain Data can't generate this view."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b712453190 = messageFormatterFn((function(  ) {
  return function(d) { return "Exclude extreme values"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b548385294 = messageFormatterFn((function(  ) {
  return function(d) { return "Extreme?"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1160444075 = messageFormatterFn((function(  ) {
  return function(d) { return "Yes"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1284039123 = messageFormatterFn((function(  ) {
  return function(d) { return "No"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a663906543 = messageFormatterFn((function(  ) {
  return function(d) { return d.columnCaption + ":"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1477965898 = messageFormatterFn((function(  ) {
  return function(d) { return "View Full Data"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1032466651 = messageFormatterFn((function(  ) {
  return function(d) { return "You can't view full data because the original mark is no longer selected or available"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b927431552 = messageFormatterFn((function(  ) {
  return function(d) { return "Previous Record"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b823662340 = messageFormatterFn((function(  ) {
  return function(d) { return "Next Record"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1247931270 = messageFormatterFn((function(  ) {
  return function(d) { return "Open in worksheet"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1659705085 = messageFormatterFn((function(  ) {
  return function(d) { return "Open in larger view"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1619057459 = messageFormatterFn((function(  ) {
  return function(d) { return "Underlying record details not available."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b249188712 = messageFormatterFn((function(  ) {
  return function(d) { return "Extreme"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a984216137 = messageFormatterFn((function(  ) {
  return function(d) { return "All Other Marks"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1055256424 = messageFormatterFn((function(  ) {
  return function(d) { return "All Marks"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1211527659 = messageFormatterFn((function(  ) {
  return function(d) { return "Other"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1316410949 = messageFormatterFn((function(  ) {
  return function(d) { return "Analyzed Mark"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1764411580 = messageFormatterFn((function(  ) {
  return function(d) { return "Average"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a60511096 = messageFormatterFn((function(  ) {
  return function(d) { return "This viz can't be analyzed for explanations."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a254071093 = messageFormatterFn((function(  ) {
  return function(d) { return "The viz must have a mark selected to be analyzed for explanations. Try selecting a single mark."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1304545467 = messageFormatterFn((function(  ) {
  return function(d) { return "Can't analyze multiple marks at the same time for explanations. Try selecting a single mark."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a497406352 = messageFormatterFn((function(  ) {
  return function(d) { return "Can't analyze this viz for explanations because the data source is not currently connected."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2055925272 = messageFormatterFn((function(  ) {
  return function(d) { return "Can't analyze data sources that don't support CountD aggregations or cross-table joins at different levels of detail."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2036003801 = messageFormatterFn((function(  ) {
  return function(d) { return "Can't analyze this viz for explanations because there are too many marks."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a303628391 = messageFormatterFn((function(  ) {
  return function(d) { return "Can't analyze a story for explanations."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b38034558 = messageFormatterFn((function(  ) {
  return function(d) { return "Can't analyze marks that aren't aggregated."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a530134888 = messageFormatterFn((function(  ) {
  return function(d) { return "Can't analyze a viz that uses a table calculation as a filter."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a981418705 = messageFormatterFn((function(  ) {
  return function(d) { return "Can't analyze a viz that uses pages."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1495000410 = messageFormatterFn((function(  ) {
  return function(d) { return "Can't analyze a viz that uses blending."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b284264670 = messageFormatterFn((function(  ) {
  return function(d) { return "Can't analyze a viz that contains clusters."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b154708451 = messageFormatterFn((function(  ) {
  return function(d) { return "Can't analyze a viz that has no dimensions, or contains only Measures Names as a dimension."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1656518884 = messageFormatterFn((function(  ) {
  return function(d) { return "Can't analyze marks that are totals."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1871239618 = messageFormatterFn((function(  ) {
  return function(d) { return "Can't analyze a viz that has no measures."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1150055075 = messageFormatterFn((function(  ) {
  return function(d) { return "Explain Data"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1454913970 = messageFormatterFn((function(  ) {
  return function(d) { return "Explain Data can't generate this view."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a273976791 = messageFormatterFn((function(  ) {
  return function(d) { return "Export As Version"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1519139174 = messageFormatterFn((function(  ) {
  return function(d) { return "This action will save a new workbook and will not overwrite your existing workbook."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b189887043 = messageFormatterFn((function(  ) {
  return function(d) { return "Export"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1966591561 = messageFormatterFn((function(  ) {
  return function(d) { return "Cancel"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a925018336 = messageFormatterFn((function(  ) {
  return function(d) { return "Export As"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b679322458 = messageFormatterFn((function(  ) {
  return function(d) { return "Learn More"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1064003278 = messageFormatterFn((function(  ) {
  return function(d) { return "You are currently running Tableau version " + d.VERSION + "."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1554450613 = messageFormatterFn((function(  ) {
  return function(d) { return "The following functionality is not available in Tableau version " + d.VERSION + ":"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1456216479 = messageFormatterFn((function(  ) {
  return function(d) { return "When you export your workbook as Tableau version " + d.VERSION + ", some functionality and visual features may be lost or degraded."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1283118476 = messageFormatterFn((function(  ) {
  return function(d) { return "Workbook functionality and features will not change when exported to Tableau " + d.VERSION + "."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1283119437 = messageFormatterFn((function(  ) {
  return function(d) { return "The new workbook has slight underlying changes but looks the same when exported to Tableau " + d.VERSION + "."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1283120398 = messageFormatterFn((function(  ) {
  return function(d) { return "The new workbook has slight underlying changes but looks the same when exported to Tableau " + d.VERSION + "."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1283121359 = messageFormatterFn((function(  ) {
  return function(d) { return "The new workbook has some feature and functionality loss when exported to Tableau " + d.VERSION + "."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1283122320 = messageFormatterFn((function(  ) {
  return function(d) { return "The new workbook has major feature and functionality loss when exported to Tableau " + d.VERSION + "."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1283123281 = messageFormatterFn((function(  ) {
  return function(d) { return "The new workbook has critical feature and functionality loss when exported to Tableau " + d.VERSION + "."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1480606447 = messageFormatterFn((function(  ) {
  return function(d) { return "CSV"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b286192475 = messageFormatterFn((function(  ) {
  return function(d) { return "Select a sheet from this dashboard"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b748361726 = messageFormatterFn((function(  ) {
  return function(d) { return "Excel"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1318823946 = messageFormatterFn((function(  ) {
  return function(d) { return "Single Sheet Selection"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1342511219 = messageFormatterFn((function(  ) {
  return function(d) { return "Select Format"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b456897196 = messageFormatterFn((function(  ) {
  return function(d) { return "Select a sheet from this storypoint"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1822007378 = messageFormatterFn((function(  ) {
  return function(d) { return "Invalid export command was found."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a217655974 = messageFormatterFn((function(  ) {
  return function(d) { return "An unexpected error occurred while generating the document. The error is 'InvalidSheetSelection'"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1607667072 = messageFormatterFn((function(  ) {
  return function(d) { return "An unexpected error occurred while rendering the export crosstab dialog. The error is 'InvalidSheetType'"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2017491017 = messageFormatterFn((function(  ) {
  return function(d) { return "An unexpected error occurred while rendering the export crosstab dialog. The error is 'InvalidFileType'"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a970374145 = messageFormatterFn((function(  ) {
  return function(d) { return "An unexpected error occurred while generating the document. More information: "; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b708589060 = messageFormatterFn((function(  ) {
  return function(d) { return "No sheets to select. Try a different view."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1033598691 = messageFormatterFn((function(  ) {
  return function(d) { return "Orientation"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a673355221 = messageFormatterFn((function(  ) {
  return function(d) { return "Portrait"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1789725737 = messageFormatterFn((function(  ) {
  return function(d) { return "Landscape"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b827901199 = messageFormatterFn((function(  ) {
  return function(d) { return "Page Size"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1423880210 = messageFormatterFn((function(  ) {
  return function(d) { return "Letter"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1767751197 = messageFormatterFn((function(  ) {
  return function(d) { return "Legal"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1256784158 = messageFormatterFn((function(  ) {
  return function(d) { return "Note"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1788311635 = messageFormatterFn((function(  ) {
  return function(d) { return "Folio"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2075595811 = messageFormatterFn((function(  ) {
  return function(d) { return "Tabloid"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a953808021 = messageFormatterFn((function(  ) {
  return function(d) { return "Ledger"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1021893735 = messageFormatterFn((function(  ) {
  return function(d) { return "Statement"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a896158674 = messageFormatterFn((function(  ) {
  return function(d) { return "Executive"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1179732969 = messageFormatterFn((function(  ) {
  return function(d) { return "A3"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a475096273 = messageFormatterFn((function(  ) {
  return function(d) { return "A4"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a469572445 = messageFormatterFn((function(  ) {
  return function(d) { return "A5"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1362599954 = messageFormatterFn((function(  ) {
  return function(d) { return "B4"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1357076126 = messageFormatterFn((function(  ) {
  return function(d) { return "B5"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1977169924 = messageFormatterFn((function(  ) {
  return function(d) { return "Quarto"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1716622241 = messageFormatterFn((function(  ) {
  return function(d) { return "Unspecified"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2069364053 = messageFormatterFn((function(  ) {
  return function(d) { return "Unable to create PDF because something went wrong. Try again."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a518448908 = messageFormatterFn((function(  ) {
  return function(d) { return "Scaling"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a193875568 = messageFormatterFn((function(  ) {
  return function(d) { return "Automatic"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a978353290 = messageFormatterFn((function(  ) {
  return function(d) { return d.PERCENT_NUMBER + "%"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1582414310 = messageFormatterFn((function(  ) {
  return function(d) { return "At most 1 page high"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a792651073 = messageFormatterFn((function(  ) {
  return function(d) { return "At most 2 pages high"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2011759319 = messageFormatterFn((function(  ) {
  return function(d) { return "At most 1 page wide"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1221996082 = messageFormatterFn((function(  ) {
  return function(d) { return "At most 2 pages wide"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1327931246 = messageFormatterFn((function(  ) {
  return function(d) { return "Clear All"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1288316526 = messageFormatterFn((function(  ) {
  return function(d) { return "Specific sheets from this dashboard"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1030980112 = messageFormatterFn((function(  ) {
  return function(d) { return "Specific sheets from this workbook"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1474662360 = messageFormatterFn((function(  ) {
  return function(d) { return "Include"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b956442154 = messageFormatterFn((function(  ) {
  return function(d) { return "Sheet Selection"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1798555195 = messageFormatterFn((function(  ) {
  return function(d) { return "Select All"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1788922267 = messageFormatterFn((function(  ) {
  return function(d) { return "This Story"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a51763151 = messageFormatterFn((function(  ) {
  return function(d) { return "This View"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1827954334 = messageFormatterFn((function(  ) {
  return function(d) { return "Sheets Selected"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b589710103 = messageFormatterFn((function(  ) {
  return function(d) { return d.x + " of " + d.y; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b985017775 = messageFormatterFn((function(  ) {
  return function(d) { return "About Extension"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b979122863 = messageFormatterFn((function(  ) {
  return function(d) { return "Version: "; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a577736630 = messageFormatterFn((function(  ) {
  return function(d) { return "OK"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1970412755 = messageFormatterFn((function(  ) {
  return function(d) { return "by " + d["0"]; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2012242316 = messageFormatterFn((function(  ) {
  return function(d) { return "Extension URL: "; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b884917292 = messageFormatterFn((function(  ) {
  return function(d) { return "Instance ID: "; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1427276885 = messageFormatterFn((function(  ) {
  return function(d) { return "Support: "; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b998334643 = messageFormatterFn((function(  ) {
  return function(d) { return "View in Gallery"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2056305683 = messageFormatterFn((function(  ) {
  return function(d) { return "Developer Website"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1710876186 = messageFormatterFn((function(  ) {
  return function(d) { return "Sandboxed"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a360131896 = messageFormatterFn((function(  ) {
  return function(d) { return "Network Enabled"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1087799557 = messageFormatterFn((function(  ) {
  return function(d) { return "Try clicking here and holding " + d.shortcutHint + "."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1947550741 = messageFormatterFn((function(  ) {
  return function(d) { return "Allow Extensions"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1784823148 = messageFormatterFn((function(  ) {
  return function(d) { return "This workbook contains the following network-enabled extensions, which are web applications that expand the capabilities of Tableau. To allow these extensions to run, select OK. "; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a86263706 = messageFormatterFn((function(  ) {
  return function(d) { return "Learn more about how extensions access your data."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a784999736 = messageFormatterFn((function(  ) {
  return function(d) { return "Extension"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1069438952 = messageFormatterFn((function(  ) {
  return function(d) { return "Created By"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1358976106 = messageFormatterFn((function(  ) {
  return function(d) { return "Full Data Access"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2051943848 = messageFormatterFn((function(  ) {
  return function(d) { return "URL"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1757104658 = messageFormatterFn((function(  ) {
  return function(d) { return "Allow Extension"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1166870441 = messageFormatterFn((function(  ) {
  return function(d) { return "Allow "; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1569575361 = messageFormatterFn((function(  ) {
  return function(d) { return "This workbook contains the following network-enabled extension, which is a web application that expands the capabilities of Tableau. To allow this extension to run, select OK."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a26978240 = messageFormatterFn((function(  ) {
  return function(d) { return "Created by:"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b393760315 = messageFormatterFn((function(  ) {
  return function(d) { return "Extension URL:"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a462320651 = messageFormatterFn((function(  ) {
  return function(d) { return "Extension Name:"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1617238258 = messageFormatterFn((function(  ) {
  return function(d) { return "Full Data Access:"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b141622681 = messageFormatterFn((function(  ) {
  return function(d) { return "Yes"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b250538867 = messageFormatterFn((function(  ) {
  return function(d) { return "No"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1442095093 = messageFormatterFn((function(  ) {
  return function(d) { return "OK"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2106885620 = messageFormatterFn((function(  ) {
  return function(d) { return "Cancel"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b933928947 = messageFormatterFn((function(  ) {
  return function(d) { return "Aggregation is not available if physical tables are used"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b753163438 = messageFormatterFn((function(  ) {
  return function(d) { return "Aggregation is not available if incremental refresh is active"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1463342405 = messageFormatterFn((function(  ) {
  return function(d) { return "Advanced Settings"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b259579148 = messageFormatterFn((function(  ) {
  return function(d) { return "Year"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1584707883 = messageFormatterFn((function(  ) {
  return function(d) { return "Quarter"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1218706177 = messageFormatterFn((function(  ) {
  return function(d) { return "Month"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a941965339 = messageFormatterFn((function(  ) {
  return function(d) { return "Day"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b736443685 = messageFormatterFn((function(  ) {
  return function(d) { return "Hour"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b153896949 = messageFormatterFn((function(  ) {
  return function(d) { return "Minute"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2137341547 = messageFormatterFn((function(  ) {
  return function(d) { return "Second"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2077976998 = messageFormatterFn((function(  ) {
  return function(d) { return "Aggregation"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1580580589 = messageFormatterFn((function(  ) {
  return function(d) { return "No"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1596407246 = messageFormatterFn((function(  ) {
  return function(d) { return "No data aggregated"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1743421077 = messageFormatterFn((function(  ) {
  return function(d) { return "Yes"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b442085951 = messageFormatterFn((function(  ) {
  return function(d) { return "Aggregate data for visible dimensions"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b484915355 = messageFormatterFn((function(  ) {
  return function(d) { return "Roll up dates to"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a492138194 = messageFormatterFn((function(  ) {
  return function(d) { return "Aggregation Uber"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1022337929 = messageFormatterFn((function(  ) {
  return function(d) { return "Incremental Refresh"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2081865795 = messageFormatterFn((function(  ) {
  return function(d) { return "All Rows: "; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a329808115 = messageFormatterFn((function(  ) {
  return function(d) { return "Create Extract"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a797903112 = messageFormatterFn((function(  ) {
  return function(d) { return "Save Settings"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b263144958 = messageFormatterFn((function(  ) {
  return function(d) { return "Cancel"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b779444412 = messageFormatterFn((function(  ) {
  return function(d) { return "OK"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a161476935 = messageFormatterFn((function(  ) {
  return function(d) { return "Learn more"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1813151067 = messageFormatterFn((function(  ) {
  return function(d) { return " about setting up data extracts."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1228808140 = messageFormatterFn((function(  ) {
  return function(d) { return "Collapse All"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1702310809 = messageFormatterFn((function(  ) {
  return function(d) { return "Column Name Dropdown"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1637617100 = messageFormatterFn((function(  ) {
  return function(d) { return "Learn more"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1788584409 = messageFormatterFn((function(  ) {
  return function(d) { return "Store data using one table for each physical table."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b234049032 = messageFormatterFn((function(  ) {
  return function(d) { return "Store data using one table for each logical table."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b957203099 = messageFormatterFn((function(  ) {
  return function(d) { return "Need help with extract settings? "; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b643070216 = messageFormatterFn((function(  ) {
  return function(d) { return "Physical Tables"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1169740033 = messageFormatterFn((function(  ) {
  return function(d) { return "Data Storage Option Dropdown"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a530521837 = messageFormatterFn((function(  ) {
  return function(d) { return "In some situations, this option can improve performance and reduce the size of your extract."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a198995300 = messageFormatterFn((function(  ) {
  return function(d) { return "Use this option if you need to use extract filters, aggregation, or other extract settings."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a48298297 = messageFormatterFn((function(  ) {
  return function(d) { return "Logical Tables"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b548315276 = messageFormatterFn((function(  ) {
  return function(d) { return "Specify how to store data in the extract"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1995995843 = messageFormatterFn((function(  ) {
  return function(d) { return "Data Storage"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b769505884 = messageFormatterFn((function(  ) {
  return function(d) { return "Hide All Unused Fields"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1049644865 = messageFormatterFn((function(  ) {
  return function(d) { return "Specify how much data to extract"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a668704880 = messageFormatterFn((function(  ) {
  return function(d) { return "Extract Data"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b427127392 = messageFormatterFn((function(  ) {
  return function(d) { return "Edit"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2114236071 = messageFormatterFn((function(  ) {
  return function(d) { return "Expand All"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a81390328 = messageFormatterFn((function(  ) {
  return function(d) { return "Edit Filter"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1206661582 = messageFormatterFn((function(  ) {
  return function(d) { return "Remove Filter"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b379242707 = messageFormatterFn((function(  ) {
  return function(d) { return "Add…"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1243120596 = messageFormatterFn((function(  ) {
  return function(d) { return "Edit"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1475370994 = messageFormatterFn((function(  ) {
  return function(d) { return "Remove"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b991601562 = messageFormatterFn((function(  ) {
  return function(d) { return "Filter Uber"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a464072565 = messageFormatterFn((function(  ) {
  return function(d) { return "Details"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1473836613 = messageFormatterFn((function(  ) {
  return function(d) { return "Filter"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2131574805 = messageFormatterFn((function(  ) {
  return function(d) { return "Filters"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1465135456 = messageFormatterFn((function(  ) {
  return function(d) { return "No filters applied"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1824765448 = messageFormatterFn((function(  ) {
  return function(d) { return "1 filter applied"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1365184347 = messageFormatterFn((function(  ) {
  return function(d) { return d.count + " filters applied"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1820756186 = messageFormatterFn((function(  ) {
  return function(d) { return "Add Filter"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1214567630 = messageFormatterFn((function(  ) {
  return function(d) { return "Hide All Unused Fields"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a789640340 = messageFormatterFn((function(  ) {
  return function(d) { return "History..."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2111321629 = messageFormatterFn((function(  ) {
  return function(d) { return "All rows will be added."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b666566235 = messageFormatterFn((function(  ) {
  return function(d) { return "Full Refresh"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1901058054 = messageFormatterFn((function(  ) {
  return function(d) { return "Identify new rows using column"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a648543247 = messageFormatterFn((function(  ) {
  return function(d) { return "Table to refresh"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a485033821 = messageFormatterFn((function(  ) {
  return function(d) { return "Enabled on field "; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a757674058 = messageFormatterFn((function(  ) {
  return function(d) { return "Not enabled"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b630277631 = messageFormatterFn((function(  ) {
  return function(d) { return "Incremental refresh is not available if data is aggregated"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1225389535 = messageFormatterFn((function(  ) {
  return function(d) { return "Incremental refresh is not available, it's disabled"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1307846717 = messageFormatterFn((function(  ) {
  return function(d) { return "Incremental refresh is not available if data does not include all rows"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1524940252 = messageFormatterFn((function(  ) {
  return function(d) { return "Incremental refresh is not available if data is aggregated"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b764263503 = messageFormatterFn((function(  ) {
  return function(d) { return "Advanced settings are not available, filters are used"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1074997857 = messageFormatterFn((function(  ) {
  return function(d) { return "Advanced settings are not available, minimum date range has been set"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1801943465 = messageFormatterFn((function(  ) {
  return function(d) { return "Minimum date range to refresh"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1327231113 = messageFormatterFn((function(  ) {
  return function(d) { return "The last " + d.count + " " + d.type + " of data from the refresh date will always be refreshed."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1708257776 = messageFormatterFn((function(  ) {
  return function(d) { return "Input must be a"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1866112585 = messageFormatterFn((function(  ) {
  return function(d) { return "positive number"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b595612942 = messageFormatterFn((function(  ) {
  return function(d) { return "Minimum date range is not available, filters are used"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2099873845 = messageFormatterFn((function(  ) {
  return function(d) { return "Minimum date range is not available, 'replace the last rows' option is selected"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b694913589 = messageFormatterFn((function(  ) {
  return function(d) { return "Date or DateTime column types are required for minimum date range refresh."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1417869049 = messageFormatterFn((function(  ) {
  return function(d) { return "second"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a585859751 = messageFormatterFn((function(  ) {
  return function(d) { return "minute"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1503429751 = messageFormatterFn((function(  ) {
  return function(d) { return "hour"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b925443329 = messageFormatterFn((function(  ) {
  return function(d) { return "day"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1923148423 = messageFormatterFn((function(  ) {
  return function(d) { return "week"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b502106397 = messageFormatterFn((function(  ) {
  return function(d) { return "month"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1252638193 = messageFormatterFn((function(  ) {
  return function(d) { return "quarter"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1980294288 = messageFormatterFn((function(  ) {
  return function(d) { return "year"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a959208493 = messageFormatterFn((function(  ) {
  return function(d) { return "unknown"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1004191454 = messageFormatterFn((function(  ) {
  return function(d) { return "seconds"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a981859202 = messageFormatterFn((function(  ) {
  return function(d) { return "minutes"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b638241870 = messageFormatterFn((function(  ) {
  return function(d) { return "hours"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1376103978 = messageFormatterFn((function(  ) {
  return function(d) { return "days"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b511864926 = messageFormatterFn((function(  ) {
  return function(d) { return "weeks"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1614646982 = messageFormatterFn((function(  ) {
  return function(d) { return "months"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b177002214 = messageFormatterFn((function(  ) {
  return function(d) { return "quarters"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1259656889 = messageFormatterFn((function(  ) {
  return function(d) { return "years"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1917789620 = messageFormatterFn((function(  ) {
  return function(d) { return "Seconds"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1343611884 = messageFormatterFn((function(  ) {
  return function(d) { return "Minutes"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1388490172 = messageFormatterFn((function(  ) {
  return function(d) { return "Hours"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1959229508 = messageFormatterFn((function(  ) {
  return function(d) { return "Days"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a579763148 = messageFormatterFn((function(  ) {
  return function(d) { return "Weeks"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1327783464 = messageFormatterFn((function(  ) {
  return function(d) { return "Months"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a982828884 = messageFormatterFn((function(  ) {
  return function(d) { return "Quarters"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1704284075 = messageFormatterFn((function(  ) {
  return function(d) { return "Years"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1889731999 = messageFormatterFn((function(  ) {
  return function(d) { return "Unknown"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1724025812 = messageFormatterFn((function(  ) {
  return function(d) { return "When new rows are retrieved:"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2028888253 = messageFormatterFn((function(  ) {
  return function(d) { return "Replace the last rows added"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b972745747 = messageFormatterFn((function(  ) {
  return function(d) { return "Delete rows in the extract with the last recorded "; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2011888807 = messageFormatterFn((function(  ) {
  return function(d) { return " value. Add rows with "; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a510445188 = messageFormatterFn((function(  ) {
  return function(d) { return " values greater than or equal to the last recorded value. Use this option if "; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1458935002 = messageFormatterFn((function(  ) {
  return function(d) { return " values are not unique for all rows of your data."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b292706396 = messageFormatterFn((function(  ) {
  return function(d) { return "Don't replace the last rows added"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1016490873 = messageFormatterFn((function(  ) {
  return function(d) { return "Add rows with "; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a600752662 = messageFormatterFn((function(  ) {
  return function(d) { return " values greater than the last recorded value."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1353208569 = messageFormatterFn((function(  ) {
  return function(d) { return "All rows"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2034201921 = messageFormatterFn((function(  ) {
  return function(d) { return "Sample"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1413273852 = messageFormatterFn((function(  ) {
  return function(d) { return "Number of Rows"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1015133068 = messageFormatterFn((function(  ) {
  return function(d) { return "Top"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b399086285 = messageFormatterFn((function(  ) {
  return function(d) { return "Sampling is not available, it's disabled"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b79338466 = messageFormatterFn((function(  ) {
  return function(d) { return "All rows must be processed when incremental refresh is active"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1857891508 = messageFormatterFn((function(  ) {
  return function(d) { return "Sample Option Dropdown"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1800973682 = messageFormatterFn((function(  ) {
  return function(d) { return "percent"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a445618118 = messageFormatterFn((function(  ) {
  return function(d) { return "rows"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1682343849 = messageFormatterFn((function(  ) {
  return function(d) { return "Sample: "; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b252959304 = messageFormatterFn((function(  ) {
  return function(d) { return "Why can I only sample one table?"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2039791447 = messageFormatterFn((function(  ) {
  return function(d) { return "Sampling Uber"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1649558146 = messageFormatterFn((function(  ) {
  return function(d) { return "Table Dropdown"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a691587454 = messageFormatterFn((function(  ) {
  return function(d) { return "Top: "; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1508842252 = messageFormatterFn((function(  ) {
  return function(d) { return "by"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a470993211 = messageFormatterFn((function(  ) {
  return function(d) { return "Input must be"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b623037101 = messageFormatterFn((function(  ) {
  return function(d) { return "number"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1599896339 = messageFormatterFn((function(  ) {
  return function(d) { return "Show template project contributors!"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a990879896 = messageFormatterFn((function(  ) {
  return function(d) { return "Hide template project contributors!"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1371172371 = messageFormatterFn((function(  ) {
  return function(d) { return "Create Extract"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a872134570 = messageFormatterFn((function(  ) {
  return function(d) { return "Data extracts might take some time to create. You can either create the extract now or wait until you select another data source or sheet, which will cause the extract to be created automatically."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b507377406 = messageFormatterFn((function(  ) {
  return function(d) { return "Importing data from:"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1517124035 = messageFormatterFn((function(  ) {
  return function(d) { return "Cancel"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b156930885 = messageFormatterFn((function(  ) {
  return function(d) { return "Create Extract"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b513806996 = messageFormatterFn((function(  ) {
  return function(d) { return "Creating Extract..."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a892863894 = messageFormatterFn((function(  ) {
  return function(d) { return "Cancel"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b883824897 = messageFormatterFn((function(  ) {
  return function(d) { return "Save"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1048744401 = messageFormatterFn((function(  ) {
  return function(d) { return "Save As..."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1190865679 = messageFormatterFn((function(  ) {
  return function(d) { return "or"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1659977996 = messageFormatterFn((function(  ) {
  return function(d) { return "and close."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a792114573 = messageFormatterFn((function(  ) {
  return function(d) { return "Elapsed Time (hh:mm:ss):"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1978957353 = messageFormatterFn((function(  ) {
  return function(d) { return "Refresh Now"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a837950016 = messageFormatterFn((function(  ) {
  return function(d) { return "Refreshing the extract may take a while. Are you sure you want to continue?"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1150879791 = messageFormatterFn((function(  ) {
  return function(d) { return "Refresh"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b433526478 = messageFormatterFn((function(  ) {
  return function(d) { return "Set Save Location"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2126215136 = messageFormatterFn((function(  ) {
  return function(d) { return "When the extract finishes, the workbook will be published at this location and you'll receive an email notification."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a477038424 = messageFormatterFn((function(  ) {
  return function(d) { return "Set Save Location"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1192464374 = messageFormatterFn((function(  ) {
  return function(d) { return "Cancel"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a444695101 = messageFormatterFn((function(  ) {
  return function(d) { return "Notify Me When Complete"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1967523479 = messageFormatterFn((function(  ) {
  return function(d) { return "Name"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2107347927 = messageFormatterFn((function(  ) {
  return function(d) { return "Project"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1519592082 = messageFormatterFn((function(  ) {
  return function(d) { return "Embed password for data source"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b836542791 = messageFormatterFn((function(  ) {
  return function(d) { return "Notify When Extract is Complete"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a971410202 = messageFormatterFn((function(  ) {
  return function(d) { return "You can now close the editor. When the extract finishes. the workbook '" + d.WORKBOOK_NAME + "' will be saved to the '" + d.PROJECT_NAME + "' project and you'll receive an email notification."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1337242201 = messageFormatterFn((function(  ) {
  return function(d) { return "Close Editor"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1927787438 = messageFormatterFn((function(  ) {
  return function(d) { return "None"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b227246645 = messageFormatterFn((function(  ) {
  return function(d) { return "Solid"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1412220768 = messageFormatterFn((function(  ) {
  return function(d) { return "Dotted"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1832359719 = messageFormatterFn((function(  ) {
  return function(d) { return "Dashed"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2083468121 = messageFormatterFn((function(  ) {
  return function(d) { return "Borders"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1603578712 = messageFormatterFn((function(  ) {
  return function(d) { return "Cell"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2042585615 = messageFormatterFn((function(  ) {
  return function(d) { return "Rows"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b132045507 = messageFormatterFn((function(  ) {
  return function(d) { return "Columns"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1972116446 = messageFormatterFn((function(  ) {
  return function(d) { return "Pane"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1050291453 = messageFormatterFn((function(  ) {
  return function(d) { return "Header"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a97560016 = messageFormatterFn((function(  ) {
  return function(d) { return "Field Labels"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1903946892 = messageFormatterFn((function(  ) {
  return function(d) { return "Row Labels"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2086665210 = messageFormatterFn((function(  ) {
  return function(d) { return "Column Labels"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2105302362 = messageFormatterFn((function(  ) {
  return function(d) { return "Horizontal"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1783243028 = messageFormatterFn((function(  ) {
  return function(d) { return "Vertical"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b132899436 = messageFormatterFn((function(  ) {
  return function(d) { return "Row Dividers"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b365491546 = messageFormatterFn((function(  ) {
  return function(d) { return "Column Dividers"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1872994404 = messageFormatterFn((function(  ) {
  return function(d) { return "Level"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b784908891 = messageFormatterFn((function(  ) {
  return function(d) { return "Totals"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1087984325 = messageFormatterFn((function(  ) {
  return function(d) { return "Grand Totals"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1281658674 = messageFormatterFn((function(  ) {
  return function(d) { return "Worksheet"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b393375541 = messageFormatterFn((function(  ) {
  return function(d) { return "Corner"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b335030012 = messageFormatterFn((function(  ) {
  return function(d) { return "pane header linked"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1538183837 = messageFormatterFn((function(  ) {
  return function(d) { return "pane header unlinked"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1914314807 = messageFormatterFn((function(  ) {
  return function(d) { return "Row Banding"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b536344457 = messageFormatterFn((function(  ) {
  return function(d) { return "Column Banding"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2065749559 = messageFormatterFn((function(  ) {
  return function(d) { return "Size"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a35676618 = messageFormatterFn((function(  ) {
  return function(d) { return "Shading"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1531209415 = messageFormatterFn((function(  ) {
  return function(d) { return "Continuous Values"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b332660545 = messageFormatterFn((function( plural, pluralFuncs ) {
  return function(d) { return plural(d.count, 0, pluralFuncs.en, { one: function() { return d.itemCount + " Row";}, other: function() { return d.itemCount + " Rows";} }); }
})(messageFormat.plural, {"en": Globalize("en").pluralGenerator()}), Globalize("en").pluralGenerator({}));
Globalize.a1045309805 = messageFormatterFn((function( plural, pluralFuncs ) {
  return function(d) { return plural(d.count, 0, pluralFuncs.en, { one: function() { return d.itemCount + " Column";}, other: function() { return d.itemCount + " Columns";} }); }
})(messageFormat.plural, {"en": Globalize("en").pluralGenerator()}), Globalize("en").pluralGenerator({}));
Globalize.a1611481766 = messageFormatterFn((function(  ) {
  return function(d) { return "Format"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2112344903 = messageFormatterFn((function(  ) {
  return function(d) { return "Uh-oh something went wrong! Please close the formatting pane and try that again."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2112343942 = messageFormatterFn((function(  ) {
  return function(d) { return "We've sent an error report to our engineers."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1035049639 = messageFormatterFn((function(  ) {
  return function(d) { return "None"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a800422450 = messageFormatterFn((function(  ) {
  return function(d) { return "Fill"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a922219986 = messageFormatterFn((function(  ) {
  return function(d) { return "Floating"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a764677631 = messageFormatterFn((function(  ) {
  return function(d) { return "Close formatting pane"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a947371056 = messageFormatterFn((function(  ) {
  return function(d) { return "Distribute Contents Evenly"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1080821406 = messageFormatterFn((function(  ) {
  return function(d) { return "HexColor"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b906452609 = messageFormatterFn((function(  ) {
  return function(d) { return "Multiple"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a968718536 = messageFormatterFn((function(  ) {
  return function(d) { return "Less"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1006539652 = messageFormatterFn((function(  ) {
  return function(d) { return "More"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1450208577 = messageFormatterFn((function(  ) {
  return function(d) { return "Interactive Controls"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1969046816 = messageFormatterFn((function(  ) {
  return function(d) { return "Lines"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2051692756 = messageFormatterFn((function(  ) {
  return function(d) { return "Legends, Filters, Highlighters, and Parameters"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b651091736 = messageFormatterFn((function(  ) {
  return function(d) { return "Font Family"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1089397525 = messageFormatterFn((function(  ) {
  return function(d) { return "Font Size"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1569929344 = messageFormatterFn((function(  ) {
  return function(d) { return "Color"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1601385635 = messageFormatterFn((function(  ) {
  return function(d) { return "Reset to Defaults"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b53890076 = messageFormatterFn((function(  ) {
  return function(d) { return "Clear " + d.context + " formatting"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1805954376 = messageFormatterFn((function(  ) {
  return function(d) { return "Borders"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1078596559 = messageFormatterFn((function(  ) {
  return function(d) { return "Shading"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b782771445 = messageFormatterFn((function(  ) {
  return function(d) { return "Fonts"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a245281674 = messageFormatterFn((function(  ) {
  return function(d) { return "Inner Padding"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a227572485 = messageFormatterFn((function(  ) {
  return function(d) { return "Outer Padding"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2036723075 = messageFormatterFn((function(  ) {
  return function(d) { return "Pane and header linked"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1701277226 = messageFormatterFn((function(  ) {
  return function(d) { return "Pane and header unlinked"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1624140734 = messageFormatterFn((function(  ) {
  return function(d) { return "Toggle between " + d.leftTab + " and " + d.rightTab; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1676521395 = messageFormatterFn((function(  ) {
  return function(d) { return "Automatic"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b275479336 = messageFormatterFn((function(  ) {
  return function(d) { return "Billions (B)"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b275474531 = messageFormatterFn((function(  ) {
  return function(d) { return "Billions (G)"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b905187885 = messageFormatterFn((function(  ) {
  return function(d) { return "Bold"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a668257826 = messageFormatterFn((function(  ) {
  return function(d) { return "Bottom Align"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2000373576 = messageFormatterFn((function(  ) {
  return function(d) { return "Center Align"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1357739140 = messageFormatterFn((function(  ) {
  return function(d) { return "Color Palette"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1489218352 = messageFormatterFn((function(  ) {
  return function(d) { return "Currency (Custom)"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1213450876 = messageFormatterFn((function(  ) {
  return function(d) { return "Currency (Standard)"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a688192255 = messageFormatterFn((function(  ) {
  return function(d) { return "Custom"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b104381866 = messageFormatterFn((function(  ) {
  return function(d) { return "Custom Format"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b860673404 = messageFormatterFn((function(  ) {
  return function(d) { return "Dark Palette"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1849409343 = messageFormatterFn((function(  ) {
  return function(d) { return "Dashed"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1121080955 = messageFormatterFn((function(  ) {
  return function(d) { return "Decimal Places"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1429270392 = messageFormatterFn((function(  ) {
  return function(d) { return "Dotted"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a961044585 = messageFormatterFn((function(  ) {
  return function(d) { return "Font Family"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1598310572 = messageFormatterFn((function(  ) {
  return function(d) { return "Font Size"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b723035712 = messageFormatterFn((function(  ) {
  return function(d) { return "Font Style"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b801346763 = messageFormatterFn((function(  ) {
  return function(d) { return "Height"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1963609501 = messageFormatterFn((function(  ) {
  return function(d) { return "Hex Value"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1853965966 = messageFormatterFn((function(  ) {
  return function(d) { return "Horizontal"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b898363810 = messageFormatterFn((function(  ) {
  return function(d) { return "Horizontal Alignment Control"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a682926459 = messageFormatterFn((function(  ) {
  return function(d) { return "Horizontal Position"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2089145022 = messageFormatterFn((function(  ) {
  return function(d) { return "Include thousands separators"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b285604499 = messageFormatterFn((function(  ) {
  return function(d) { return "is Off"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b701944073 = messageFormatterFn((function(  ) {
  return function(d) { return "is On"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1144714750 = messageFormatterFn((function(  ) {
  return function(d) { return "Italic"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b834892351 = messageFormatterFn((function(  ) {
  return function(d) { return "Learn more"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1802074118 = messageFormatterFn((function(  ) {
  return function(d) { return "Left Align"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2012091724 = messageFormatterFn((function(  ) {
  return function(d) { return "Level"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2112852574 = messageFormatterFn((function(  ) {
  return function(d) { return "Light Palette"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a159428069 = messageFormatterFn((function(  ) {
  return function(d) { return "Line Style"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1842066384 = messageFormatterFn((function(  ) {
  return function(d) { return "Line Width"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1879947720 = messageFormatterFn((function(  ) {
  return function(d) { return "Linked. Click to unlink."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2101710897 = messageFormatterFn((function(  ) {
  return function(d) { return "Unlinked. Click to link."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1982818216 = messageFormatterFn((function(  ) {
  return function(d) { return "Locale"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1433189512 = messageFormatterFn((function(  ) {
  return function(d) { return "Middle Align"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a184768725 = messageFormatterFn((function(  ) {
  return function(d) { return "Millions (M)"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2146684101 = messageFormatterFn((function(  ) {
  return function(d) { return "Negative Values"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1310885868 = messageFormatterFn((function(  ) {
  return function(d) { return "None"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b561577530 = messageFormatterFn((function(  ) {
  return function(d) { return "None"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1785190184 = messageFormatterFn((function(  ) {
  return function(d) { return "Number (Custom)"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2120427404 = messageFormatterFn((function(  ) {
  return function(d) { return "Number (Standard)"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1457503153 = messageFormatterFn((function(  ) {
  return function(d) { return "Number Type"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a233245907 = messageFormatterFn((function(  ) {
  return function(d) { return "Opacity"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a513257173 = messageFormatterFn((function(  ) {
  return function(d) { return "Bottom Inner Padding"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1030871512 = messageFormatterFn((function(  ) {
  return function(d) { return "Bottom inner padding stepper widget"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b539340480 = messageFormatterFn((function(  ) {
  return function(d) { return "Inner Padding"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1729233017 = messageFormatterFn((function(  ) {
  return function(d) { return "Left Inner Padding"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2120421708 = messageFormatterFn((function(  ) {
  return function(d) { return "Left inner padding stepper widget"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1026160884 = messageFormatterFn((function(  ) {
  return function(d) { return "Right Inner Padding"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b442234087 = messageFormatterFn((function(  ) {
  return function(d) { return "Right inner padding stepper widget"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a488309325 = messageFormatterFn((function(  ) {
  return function(d) { return "Inner padding stepper widget"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a695995483 = messageFormatterFn((function(  ) {
  return function(d) { return "Top Inner Padding"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b127336430 = messageFormatterFn((function(  ) {
  return function(d) { return "Top inner padding stepper widget"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b569099971 = messageFormatterFn((function(  ) {
  return function(d) { return "Padding is linked. Click to unlink."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a891197584 = messageFormatterFn((function(  ) {
  return function(d) { return "Bottom Outer Padding"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a586075389 = messageFormatterFn((function(  ) {
  return function(d) { return "Bottom outer padding stepper widget"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b111811781 = messageFormatterFn((function(  ) {
  return function(d) { return "Outer Padding"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1412308212 = messageFormatterFn((function(  ) {
  return function(d) { return "Left Outer Padding"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b6920423 = messageFormatterFn((function(  ) {
  return function(d) { return "Left outer padding stepper widget"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b208573479 = messageFormatterFn((function(  ) {
  return function(d) { return "Right Outer Padding"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a651796308 = messageFormatterFn((function(  ) {
  return function(d) { return "Right padding outer stepper widget"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b746425038 = messageFormatterFn((function(  ) {
  return function(d) { return "Outer padding stepper widget"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1669532544 = messageFormatterFn((function(  ) {
  return function(d) { return "Top Outer Padding"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1998821619 = messageFormatterFn((function(  ) {
  return function(d) { return "Top outer padding stepper widget"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1994463740 = messageFormatterFn((function(  ) {
  return function(d) { return "Padding is unlinked. Click to link."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b749249944 = messageFormatterFn((function(  ) {
  return function(d) { return "Percentage"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a885748582 = messageFormatterFn((function(  ) {
  return function(d) { return "Position"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1206526496 = messageFormatterFn((function(  ) {
  return function(d) { return "Prefix"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1493110359 = messageFormatterFn((function(  ) {
  return function(d) { return "Right Align"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a610846775 = messageFormatterFn((function(  ) {
  return function(d) { return "Scientific"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1008073700 = messageFormatterFn((function(  ) {
  return function(d) { return "Show Color Picker"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1828315582 = messageFormatterFn((function(  ) {
  return function(d) { return "Size"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b88149325 = messageFormatterFn((function(  ) {
  return function(d) { return "Solid"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1876889889 = messageFormatterFn((function(  ) {
  return function(d) { return "Suffix"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b118037204 = messageFormatterFn((function(  ) {
  return function(d) { return "Text Alignment"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b33518022 = messageFormatterFn((function(  ) {
  return function(d) { return "Auto"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a46916717 = messageFormatterFn((function(  ) {
  return function(d) { return "Down"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b455887713 = messageFormatterFn((function(  ) {
  return function(d) { return "Text Orientation is " + d.textOrientationSelection; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2051617934 = messageFormatterFn((function(  ) {
  return function(d) { return "Normal"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b674306010 = messageFormatterFn((function(  ) {
  return function(d) { return "Up"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b802429859 = messageFormatterFn((function(  ) {
  return function(d) { return "Thousands (K)"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b197163586 = messageFormatterFn((function(  ) {
  return function(d) { return "Top Align"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a526557204 = messageFormatterFn((function(  ) {
  return function(d) { return "Underline"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1655800439 = messageFormatterFn((function(  ) {
  return function(d) { return "Display Units"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1716543748 = messageFormatterFn((function(  ) {
  return function(d) { return "Vertical"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b951765492 = messageFormatterFn((function(  ) {
  return function(d) { return "Vertical Alignment Control"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b805923059 = messageFormatterFn((function(  ) {
  return function(d) { return "Vertical Position"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1011933426 = messageFormatterFn((function(  ) {
  return function(d) { return "Width"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2029951215 = messageFormatterFn((function(  ) {
  return function(d) { return "Auto"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2079192744 = messageFormatterFn((function(  ) {
  return function(d) { return "Text Wrap is " + d.wordWrapSelection; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a224079011 = messageFormatterFn((function(  ) {
  return function(d) { return "Wrap Off"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1946896641 = messageFormatterFn((function(  ) {
  return function(d) { return "Wrap On"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b702999964 = messageFormatterFn((function(  ) {
  return function(d) { return "Go to"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2069928378 = messageFormatterFn((function(  ) {
  return function(d) { return "Type here to filter list"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1391068136 = messageFormatterFn((function(  ) {
  return function(d) { return "Please click on an item in the list to jump directly to it"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1517151008 = messageFormatterFn((function(  ) {
  return function(d) { return "Show template project contributors!"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2126167451 = messageFormatterFn((function(  ) {
  return function(d) { return "Hide template project contributors!"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2015070063 = messageFormatterFn((function(  ) {
  return function(d) { return "Copy"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1757874750 = messageFormatterFn((function(  ) {
  return function(d) { return "Load"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1711042466 = messageFormatterFn((function(  ) {
  return function(d) { return "Objects"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a327997874 = messageFormatterFn((function(  ) {
  return function(d) { return "Tiled"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a361195430 = messageFormatterFn((function(  ) {
  return function(d) { return "Floating"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a564568603 = messageFormatterFn((function(  ) {
  return function(d) { return "Show dashboard title"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1193270721 = messageFormatterFn((function(  ) {
  return function(d) { return "Center Image"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a308514941 = messageFormatterFn((function(  ) {
  return function(d) { return "Edit Image Object"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1804510315 = messageFormatterFn((function(  ) {
  return function(d) { return "Fit Image"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1462883740 = messageFormatterFn((function(  ) {
  return function(d) { return "Choose an image file..."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b129089618 = messageFormatterFn((function(  ) {
  return function(d) { return "Choose"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b521992972 = messageFormatterFn((function(  ) {
  return function(d) { return "Options"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b825731590 = messageFormatterFn((function(  ) {
  return function(d) { return "OK"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a365438392 = messageFormatterFn((function(  ) {
  return function(d) { return "Cancel"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b500365626 = messageFormatterFn((function(  ) {
  return function(d) { return "Apply"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a844604035 = messageFormatterFn((function(  ) {
  return function(d) { return "URL Opened When Image Is Clicked"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1034048028 = messageFormatterFn((function(  ) {
  return function(d) { return "Alt Text"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a25568513 = messageFormatterFn((function(  ) {
  return function(d) { return "Insert Image File"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a447021848 = messageFormatterFn((function(  ) {
  return function(d) { return "Link to Image"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a42116916 = messageFormatterFn((function(  ) {
  return function(d) { return "File or Link Toggle"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1282810502 = messageFormatterFn((function(  ) {
  return function(d) { return "Enter URL for web image..."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1174671865 = messageFormatterFn((function(  ) {
  return function(d) { return "OK"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a793025719 = messageFormatterFn((function(  ) {
  return function(d) { return "Cancel"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a44325061 = messageFormatterFn((function(  ) {
  return function(d) { return "Duplicate Map Name"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1803254730 = messageFormatterFn((function(  ) {
  return function(d) { return "This workbook already uses a map named \"" + d.mapName + "\""; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1872989789 = messageFormatterFn((function(  ) {
  return function(d) { return "This name is already in use."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1882367398 = messageFormatterFn((function(  ) {
  return function(d) { return "Discard new map"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1603540448 = messageFormatterFn((function(  ) {
  return function(d) { return "Replace existing map with new map"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a334661023 = messageFormatterFn((function(  ) {
  return function(d) { return "Rename new map"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1947582368 = messageFormatterFn((function(  ) {
  return function(d) { return "New map name"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b905027829 = messageFormatterFn((function(  ) {
  return function(d) { return "\"" + d.mapName + "\" is a reserved name for Tableau maps"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b528528390 = messageFormatterFn((function(  ) {
  return function(d) { return "OK"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b80364078 = messageFormatterFn((function(  ) {
  return function(d) { return "Show template project contributors!"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b689380521 = messageFormatterFn((function(  ) {
  return function(d) { return "Hide template project contributors!"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1769463081 = messageFormatterFn((function(  ) {
  return function(d) { return "Select Schema Levels"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1054400938 = messageFormatterFn((function(  ) {
  return function(d) { return "Select the schema levels you want to make available as dimensions and measures in the worksheet."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a96257163 = messageFormatterFn((function(  ) {
  return function(d) { return "Schema"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a337864785 = messageFormatterFn((function(  ) {
  return function(d) { return "Example Value"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1256140371 = messageFormatterFn((function(  ) {
  return function(d) { return "schema level"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2018362038 = messageFormatterFn((function(  ) {
  return function(d) { return "schema levels"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2120279627 = messageFormatterFn((function(  ) {
  return function(d) { return "selected. This high number may slow load time."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a308411830 = messageFormatterFn((function(  ) {
  return function(d) { return "Show Fields"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1736654190 = messageFormatterFn((function(  ) {
  return function(d) { return "selected"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b334179541 = messageFormatterFn((function(  ) {
  return function(d) { return "OK"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a527213769 = messageFormatterFn((function(  ) {
  return function(d) { return "Cancel"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2142575742 = messageFormatterFn((function(  ) {
  return function(d) { return "Select All"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1314370083 = messageFormatterFn((function(  ) {
  return function(d) { return "Deselect All"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1962754627 = messageFormatterFn((function(  ) {
  return function(d) { return "Schema is inferred from scanning a sample of the document."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1132767791 = messageFormatterFn((function(  ) {
  return function(d) { return "Don’t see the data you need?"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1234702005 = messageFormatterFn((function(  ) {
  return function(d) { return "Tableau Desktop offers a rich experience to scan entire JSON files. You can continue your flow in Tableau Desktop for further analysis."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1051889257 = messageFormatterFn((function(  ) {
  return function(d) { return "⌃"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b789717448 = messageFormatterFn((function(  ) {
  return function(d) { return "Ctrl+"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b32342593 = messageFormatterFn((function(  ) {
  return function(d) { return "⌥"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1663098756 = messageFormatterFn((function(  ) {
  return function(d) { return "Alt+"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1647124622 = messageFormatterFn((function(  ) {
  return function(d) { return "⇧"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b817358819 = messageFormatterFn((function(  ) {
  return function(d) { return "Shift+"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b611620386 = messageFormatterFn((function(  ) {
  return function(d) { return "Win+"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1893362522 = messageFormatterFn((function(  ) {
  return function(d) { return "⌘"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1210531364 = messageFormatterFn((function(  ) {
  return function(d) { return "A"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1210530403 = messageFormatterFn((function(  ) {
  return function(d) { return "B"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1210529442 = messageFormatterFn((function(  ) {
  return function(d) { return "C"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1210528481 = messageFormatterFn((function(  ) {
  return function(d) { return "D"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1210527520 = messageFormatterFn((function(  ) {
  return function(d) { return "E"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1210526559 = messageFormatterFn((function(  ) {
  return function(d) { return "F"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1210525598 = messageFormatterFn((function(  ) {
  return function(d) { return "G"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1210524637 = messageFormatterFn((function(  ) {
  return function(d) { return "H"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1210523676 = messageFormatterFn((function(  ) {
  return function(d) { return "I"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1210522715 = messageFormatterFn((function(  ) {
  return function(d) { return "J"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1210521754 = messageFormatterFn((function(  ) {
  return function(d) { return "K"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1210520793 = messageFormatterFn((function(  ) {
  return function(d) { return "L"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1210519832 = messageFormatterFn((function(  ) {
  return function(d) { return "M"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1210518871 = messageFormatterFn((function(  ) {
  return function(d) { return "N"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1210517910 = messageFormatterFn((function(  ) {
  return function(d) { return "O"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1210516949 = messageFormatterFn((function(  ) {
  return function(d) { return "P"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1210515988 = messageFormatterFn((function(  ) {
  return function(d) { return "Q"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1210515027 = messageFormatterFn((function(  ) {
  return function(d) { return "R"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1210514066 = messageFormatterFn((function(  ) {
  return function(d) { return "S"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1210513105 = messageFormatterFn((function(  ) {
  return function(d) { return "T"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1210512144 = messageFormatterFn((function(  ) {
  return function(d) { return "U"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1210511183 = messageFormatterFn((function(  ) {
  return function(d) { return "V"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1210510222 = messageFormatterFn((function(  ) {
  return function(d) { return "W"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1210509261 = messageFormatterFn((function(  ) {
  return function(d) { return "X"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1210508300 = messageFormatterFn((function(  ) {
  return function(d) { return "Y"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1210507339 = messageFormatterFn((function(  ) {
  return function(d) { return "Z"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1151432803 = messageFormatterFn((function(  ) {
  return function(d) { return "0"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a645688231 = messageFormatterFn((function(  ) {
  return function(d) { return "1"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a650583565 = messageFormatterFn((function(  ) {
  return function(d) { return "2"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2015886175 = messageFormatterFn((function(  ) {
  return function(d) { return "3"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1714688357 = messageFormatterFn((function(  ) {
  return function(d) { return "4"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1720212185 = messageFormatterFn((function(  ) {
  return function(d) { return "5"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a649251619 = messageFormatterFn((function(  ) {
  return function(d) { return "6"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1046197774 = messageFormatterFn((function(  ) {
  return function(d) { return "7"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1606807056 = messageFormatterFn((function(  ) {
  return function(d) { return "8"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1491417305 = messageFormatterFn((function(  ) {
  return function(d) { return "9"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a436141888 = messageFormatterFn((function(  ) {
  return function(d) { return "F1"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a436142849 = messageFormatterFn((function(  ) {
  return function(d) { return "F2"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a436143810 = messageFormatterFn((function(  ) {
  return function(d) { return "F3"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a436144771 = messageFormatterFn((function(  ) {
  return function(d) { return "F4"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a436145732 = messageFormatterFn((function(  ) {
  return function(d) { return "F5"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a436146693 = messageFormatterFn((function(  ) {
  return function(d) { return "F6"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a436147654 = messageFormatterFn((function(  ) {
  return function(d) { return "F7"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a436148615 = messageFormatterFn((function(  ) {
  return function(d) { return "F8"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a436149576 = messageFormatterFn((function(  ) {
  return function(d) { return "F9"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a635508358 = messageFormatterFn((function(  ) {
  return function(d) { return "F10"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a635509319 = messageFormatterFn((function(  ) {
  return function(d) { return "F11"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a635510280 = messageFormatterFn((function(  ) {
  return function(d) { return "F12"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1761867289 = messageFormatterFn((function(  ) {
  return function(d) { return "Enter"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1506255486 = messageFormatterFn((function(  ) {
  return function(d) { return "↵"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1796047699 = messageFormatterFn((function(  ) {
  return function(d) { return "/"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1213899343 = messageFormatterFn((function(  ) {
  return function(d) { return "↓"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1661496268 = messageFormatterFn((function(  ) {
  return function(d) { return "Down arrow"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b994602026 = messageFormatterFn((function(  ) {
  return function(d) { return "←"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1880793585 = messageFormatterFn((function(  ) {
  return function(d) { return "Left arrow"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a377322755 = messageFormatterFn((function(  ) {
  return function(d) { return "→"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b679726520 = messageFormatterFn((function(  ) {
  return function(d) { return "Right arrow"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b58858902 = messageFormatterFn((function(  ) {
  return function(d) { return "↑"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a770947909 = messageFormatterFn((function(  ) {
  return function(d) { return "Up arrow"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b432852224 = messageFormatterFn((function(  ) {
  return function(d) { return "Backspace"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1550509729 = messageFormatterFn((function(  ) {
  return function(d) { return "Caps lock"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a9246422 = messageFormatterFn((function(  ) {
  return function(d) { return "Comma"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b246365381 = messageFormatterFn((function(  ) {
  return function(d) { return ","; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1523725164 = messageFormatterFn((function(  ) {
  return function(d) { return "Menu"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1708958560 = messageFormatterFn((function(  ) {
  return function(d) { return "Del"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a636452060 = messageFormatterFn((function(  ) {
  return function(d) { return "End"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1470654580 = messageFormatterFn((function(  ) {
  return function(d) { return "="; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1431166449 = messageFormatterFn((function(  ) {
  return function(d) { return "Plus sign"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b331033334 = messageFormatterFn((function(  ) {
  return function(d) { return "+"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1564524362 = messageFormatterFn((function(  ) {
  return function(d) { return "Esc"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b898555663 = messageFormatterFn((function(  ) {
  return function(d) { return "⎋"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1657680876 = messageFormatterFn((function(  ) {
  return function(d) { return "Home"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1431033070 = messageFormatterFn((function(  ) {
  return function(d) { return "Insert"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1227619198 = messageFormatterFn((function(  ) {
  return function(d) { return "/"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b601872153 = messageFormatterFn((function(  ) {
  return function(d) { return "*"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1784945145 = messageFormatterFn((function(  ) {
  return function(d) { return "Minus sign"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b684812030 = messageFormatterFn((function(  ) {
  return function(d) { return "-"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a721826182 = messageFormatterFn((function(  ) {
  return function(d) { return "Page down"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1748739967 = messageFormatterFn((function(  ) {
  return function(d) { return "Page up"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1323165174 = messageFormatterFn((function(  ) {
  return function(d) { return "Period"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1989133873 = messageFormatterFn((function(  ) {
  return function(d) { return "."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1341656263 = messageFormatterFn((function(  ) {
  return function(d) { return "Spacebar"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1086044460 = messageFormatterFn((function(  ) {
  return function(d) { return "␣"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a649915670 = messageFormatterFn((function(  ) {
  return function(d) { return "Tab"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2038944409 = messageFormatterFn((function(  ) {
  return function(d) { return "~"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1100111879 = messageFormatterFn((function(  ) {
  return function(d) { return "`"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a575931931 = messageFormatterFn((function(  ) {
  return function(d) { return "!"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a490734853 = messageFormatterFn((function(  ) {
  return function(d) { return "@"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1330686771 = messageFormatterFn((function(  ) {
  return function(d) { return "#"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1130080718 = messageFormatterFn((function(  ) {
  return function(d) { return "$"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1306444771 = messageFormatterFn((function(  ) {
  return function(d) { return "%"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b387164156 = messageFormatterFn((function(  ) {
  return function(d) { return "^"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1389384074 = messageFormatterFn((function(  ) {
  return function(d) { return "&"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b319168915 = messageFormatterFn((function(  ) {
  return function(d) { return "("; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b591022157 = messageFormatterFn((function(  ) {
  return function(d) { return ")"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1214674001 = messageFormatterFn((function(  ) {
  return function(d) { return "_"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a161654282 = messageFormatterFn((function(  ) {
  return function(d) { return "Open brace"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1350855632 = messageFormatterFn((function(  ) {
  return function(d) { return "Close brace"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b218490206 = messageFormatterFn((function(  ) {
  return function(d) { return "["; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a523549288 = messageFormatterFn((function(  ) {
  return function(d) { return "]"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b460137484 = messageFormatterFn((function(  ) {
  return function(d) { return "<"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b297653394 = messageFormatterFn((function(  ) {
  return function(d) { return ">"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b368172024 = messageFormatterFn((function(  ) {
  return function(d) { return "?"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a463133397 = messageFormatterFn((function(  ) {
  return function(d) { return "'"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1662449876 = messageFormatterFn((function(  ) {
  return function(d) { return "\""; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a8394976 = messageFormatterFn((function(  ) {
  return function(d) { return ":"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1174068914 = messageFormatterFn((function(  ) {
  return function(d) { return ";"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a866172478 = messageFormatterFn((function(  ) {
  return function(d) { return "|"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b546889289 = messageFormatterFn((function(  ) {
  return function(d) { return "Backslash"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a510556056 = messageFormatterFn((function(  ) {
  return function(d) { return "Show template project contributors!"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b98460387 = messageFormatterFn((function(  ) {
  return function(d) { return "Hide template project contributors!"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1939011345 = messageFormatterFn((function(  ) {
  return function(d) { return "Analytics Extensions Connections"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2013161111 = messageFormatterFn((function(  ) {
  return function(d) { return "Choose an Analytics Extensions Connection"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1765732667 = messageFormatterFn((function(  ) {
  return function(d) { return "Choose a connection"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1466655230 = messageFormatterFn((function(  ) {
  return function(d) { return "If no connection is selected, some vizzes in the published workbook may break. Please contact your server admin for any connection related issue."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a219575801 = messageFormatterFn((function(  ) {
  return function(d) { return "No Connection"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b583699905 = messageFormatterFn((function(  ) {
  return function(d) { return "Select..."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1375119597 = messageFormatterFn((function(  ) {
  return function(d) { return "Binding Connection Failed"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1860964696 = messageFormatterFn((function(  ) {
  return function(d) { return "Choose Connection"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1898445763 = messageFormatterFn((function(  ) {
  return function(d) { return "Cancel"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2115637335 = messageFormatterFn((function(  ) {
  return function(d) { return "Manage Viz Extensions"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a348185161 = messageFormatterFn((function(  ) {
  return function(d) { return "Cancel"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a684982867 = messageFormatterFn((function(  ) {
  return function(d) { return "Remove"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a150197369 = messageFormatterFn((function(  ) {
  return function(d) { return "remove selected viz extensions"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1644858736 = messageFormatterFn((function(  ) {
  return function(d) { return "Select Extensions that you want to remove"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b198230809 = messageFormatterFn((function(  ) {
  return function(d) { return "No viz extensions in workbook."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a462807034 = messageFormatterFn((function(  ) {
  return function(d) { return "Created By"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1804258135 = messageFormatterFn((function(  ) {
  return function(d) { return "Description"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a310086963 = messageFormatterFn((function(  ) {
  return function(d) { return "Version"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1333531447 = messageFormatterFn((function(  ) {
  return function(d) { return "Full Data Access"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a673004768 = messageFormatterFn((function(  ) {
  return function(d) { return "Yes"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a852673278 = messageFormatterFn((function(  ) {
  return function(d) { return "No"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b491060949 = messageFormatterFn((function(  ) {
  return function(d) { return "Extension Support"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1876009012 = messageFormatterFn((function(  ) {
  return function(d) { return "Developer website"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b305278133 = messageFormatterFn((function(  ) {
  return function(d) { return "Extension URL"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1220743325 = messageFormatterFn((function(  ) {
  return function(d) { return "The following extensions you have selected are being used in this workbook. Removing them will change vizzes in this worksheets to the Automatic mark type."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1051787990 = messageFormatterFn((function(  ) {
  return function(d) { return "Details"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1031334202 = messageFormatterFn((function(  ) {
  return function(d) { return "Viz Extensions in Use"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a852668473 = messageFormatterFn((function(  ) {
  return function(d) { return "OK"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1897987731 = messageFormatterFn((function(  ) {
  return function(d) { return "Viz Extension"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b677314852 = messageFormatterFn((function(  ) {
  return function(d) { return "Worksheets"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b308253945 = messageFormatterFn((function(  ) {
  return function(d) { return "Dashboards"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a563745089 = messageFormatterFn((function(  ) {
  return function(d) { return "Stories"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b917460427 = messageFormatterFn((function(  ) {
  return function(d) { return "none"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a593931126 = messageFormatterFn((function(  ) {
  return function(d) { return "Layer Control"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b834120315 = messageFormatterFn((function(  ) {
  return function(d) { return "Settings"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b117365853 = messageFormatterFn((function(  ) {
  return function(d) { return "Reset"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1203029983 = messageFormatterFn((function(  ) {
  return function(d) { return "Marks"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a437805073 = messageFormatterFn((function(  ) {
  return function(d) { return "Demographics"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a901988625 = messageFormatterFn((function(  ) {
  return function(d) { return "Background"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a214381833 = messageFormatterFn((function(  ) {
  return function(d) { return "An error occurred when changing the map style. Please try again."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1897384846 = messageFormatterFn((function(  ) {
  return function(d) { return "Disable interaction"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1175010809 = messageFormatterFn((function(  ) {
  return function(d) { return "Enable interaction"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1262455959 = messageFormatterFn((function(  ) {
  return function(d) { return "Hide " + d.layerName; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1456315184 = messageFormatterFn((function(  ) {
  return function(d) { return "Unhide " + d.layerName; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1311812130 = messageFormatterFn((function(  ) {
  return function(d) { return "Background"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a141862873 = messageFormatterFn((function(  ) {
  return function(d) { return "Close"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1015908962 = messageFormatterFn((function(  ) {
  return function(d) { return "Data Layer"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a468102643 = messageFormatterFn((function(  ) {
  return function(d) { return "Layer"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1875009747 = messageFormatterFn((function(  ) {
  return function(d) { return "Detail"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2291785 = messageFormatterFn((function(  ) {
  return function(d) { return "Palette"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b541350664 = messageFormatterFn((function(  ) {
  return function(d) { return "Zoom in to see this layer."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1015499361 = messageFormatterFn((function(  ) {
  return function(d) { return "Category " + d.category; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1948960405 = messageFormatterFn((function(  ) {
  return function(d) { return "Background Map Layers"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1724464780 = messageFormatterFn((function(  ) {
  return function(d) { return "Make Default"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b836153298 = messageFormatterFn((function(  ) {
  return function(d) { return "An error occurred loading the Map Layers pane. Reopen the pane and try again."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1611602780 = messageFormatterFn((function(  ) {
  return function(d) { return "An error occurred when changing the map style. Please try again."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1559148941 = messageFormatterFn((function(  ) {
  return function(d) { return "No layers are available."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a42425948 = messageFormatterFn((function(  ) {
  return function(d) { return "Background Layers"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a671684513 = messageFormatterFn((function(  ) {
  return function(d) { return "Repeat Background"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a806400744 = messageFormatterFn((function(  ) {
  return function(d) { return "Reset to defaults"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a542242575 = messageFormatterFn((function(  ) {
  return function(d) { return "Style"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1550644411 = messageFormatterFn((function(  ) {
  return function(d) { return "Washout (%)"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1458174053 = messageFormatterFn((function(  ) {
  return function(d) { return "Add Mapbox Map"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1575972907 = messageFormatterFn((function(  ) {
  return function(d) { return "Add WMS Map"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1859793129 = messageFormatterFn((function(  ) {
  return function(d) { return "Cancel"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1083526173 = messageFormatterFn((function(  ) {
  return function(d) { return "Close"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1774172008 = messageFormatterFn((function(  ) {
  return function(d) { return "Edit Mapbox Map"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1812489346 = messageFormatterFn((function(  ) {
  return function(d) { return "Edit WMS Map"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b694180507 = messageFormatterFn((function(  ) {
  return function(d) { return "Error"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a78028830 = messageFormatterFn((function(  ) {
  return function(d) { return "Add Map"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2032169173 = messageFormatterFn((function(  ) {
  return function(d) { return "OK"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b598900998 = messageFormatterFn((function(  ) {
  return function(d) { return "Map Service Type"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1863983692 = messageFormatterFn((function(  ) {
  return function(d) { return "Mapbox"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a173315285 = messageFormatterFn((function(  ) {
  return function(d) { return "An error occurred when adding a new Mapbox connection. Verify that the Mapbox URL is correct and try again"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b43436726 = messageFormatterFn((function(  ) {
  return function(d) { return "An error occurred when editing the Mapbox connection. Verify that the Mapbox URL is correct and try again"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1015133411 = messageFormatterFn((function(  ) {
  return function(d) { return "A problem was encountered while connecting to the Mapbox server. Please verify your API token and Mapbox map IDs."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1464165057 = messageFormatterFn((function(  ) {
  return function(d) { return "Learn more"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1591782159 = messageFormatterFn((function(  ) {
  return function(d) { return "The URL entered is not in a valid format."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1487694666 = messageFormatterFn((function(  ) {
  return function(d) { return "This style name is already in use."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a167142293 = messageFormatterFn((function(  ) {
  return function(d) { return "Style Name"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1836503508 = messageFormatterFn((function(  ) {
  return function(d) { return "Enter Style Name"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1024117726 = messageFormatterFn((function(  ) {
  return function(d) { return "URL"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a585647265 = messageFormatterFn((function(  ) {
  return function(d) { return "Enter URL"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1718916554 = messageFormatterFn((function(  ) {
  return function(d) { return "WMS"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a425686591 = messageFormatterFn((function(  ) {
  return function(d) { return "An error occurred when adding a new WMS connection. Verify that the WMS URL is correct and try again"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1915722545 = messageFormatterFn((function(  ) {
  return function(d) { return "Use tiled maps (recommended)"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b809860832 = messageFormatterFn((function(  ) {
  return function(d) { return "An error occurred when editing the WMS connection. Verify that the WMS URL is correct and try again"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1646482000 = messageFormatterFn((function(  ) {
  return function(d) { return "Add..."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b497101788 = messageFormatterFn((function(  ) {
  return function(d) { return "Are you sure you want to delete this map?"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b74110617 = messageFormatterFn((function(  ) {
  return function(d) { return "OK"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b25146075 = messageFormatterFn((function(  ) {
  return function(d) { return "Cancel"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b146873514 = messageFormatterFn((function(  ) {
  return function(d) { return "Delete"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1701277479 = messageFormatterFn((function(  ) {
  return function(d) { return "Confirm Deletion"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1589337786 = messageFormatterFn((function(  ) {
  return function(d) { return "URL / Description"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a7281075 = messageFormatterFn((function(  ) {
  return function(d) { return "Download"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1059896373 = messageFormatterFn((function(  ) {
  return function(d) { return "Edit"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1140545631 = messageFormatterFn((function(  ) {
  return function(d) { return "Export"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1949223088 = messageFormatterFn((function(  ) {
  return function(d) { return "Import..."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b58022066 = messageFormatterFn((function(  ) {
  return function(d) { return "There was a problem importing the map. Try again, or try connecting to a different file."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b428734383 = messageFormatterFn((function(  ) {
  return function(d) { return "Name"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1828917021 = messageFormatterFn((function(  ) {
  return function(d) { return "Map: " + d.mapSourceName + " is being used by these worksheets:"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1652403847 = messageFormatterFn((function(  ) {
  return function(d) { return "Manage Maps"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1716868077 = messageFormatterFn((function(  ) {
  return function(d) { return "Advanced"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b557455757 = messageFormatterFn((function(  ) {
  return function(d) { return "The username or password is not valid.  Check the database name and credentials and try again."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2019210155 = messageFormatterFn((function(  ) {
  return function(d) { return "Browse"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b372809876 = messageFormatterFn((function(  ) {
  return function(d) { return "Browse..."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1801191185 = messageFormatterFn((function(  ) {
  return function(d) { return "Can’t connect to " + d.dataSourceCaption; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a32281146 = messageFormatterFn((function(  ) {
  return function(d) { return "Detailed Error Message"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1677857790 = messageFormatterFn((function(  ) {
  return function(d) { return "To use this connector, you need to download and install the driver first."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1485987045 = messageFormatterFn((function(  ) {
  return function(d) { return "Download Driver"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1203842669 = messageFormatterFn((function(  ) {
  return function(d) { return "General"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1745541557 = messageFormatterFn((function(  ) {
  return function(d) { return "Initial SQL"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1861609276 = messageFormatterFn((function(  ) {
  return function(d) { return "Insert"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1439877156 = messageFormatterFn((function(  ) {
  return function(d) { return "Learn More"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1934862022 = messageFormatterFn((function(  ) {
  return function(d) { return "No file chosen"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1890820181 = messageFormatterFn((function(  ) {
  return function(d) { return "Optional"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2092391539 = messageFormatterFn((function(  ) {
  return function(d) { return "Sign In"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a320521499 = messageFormatterFn((function(  ) {
  return function(d) { return "Information and Support"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1623114143 = messageFormatterFn((function(  ) {
  return function(d) { return "For support, contact"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b518975062 = messageFormatterFn((function(  ) {
  return function(d) { return "Current Version"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1510593862 = messageFormatterFn((function(  ) {
  return function(d) { return "Support"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a112484844 = messageFormatterFn((function(  ) {
  return function(d) { return "Update and Restart"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a588698153 = messageFormatterFn((function(  ) {
  return function(d) { return "There is an update available for this connector."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a270282909 = messageFormatterFn((function(  ) {
  return function(d) { return "Name"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1290834240 = messageFormatterFn((function(  ) {
  return function(d) { return "Description"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1307179255 = messageFormatterFn((function(  ) {
  return function(d) { return "Owner"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a405642715 = messageFormatterFn((function(  ) {
  return function(d) { return "Modified"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1784175371 = messageFormatterFn((function(  ) {
  return function(d) { return "Hidden User"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b417186555 = messageFormatterFn((function(  ) {
  return function(d) { return "Currency"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1196659487 = messageFormatterFn((function(  ) {
  return function(d) { return "Location"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b498014892 = messageFormatterFn((function(  ) {
  return function(d) { return "Please publish a data role before applying."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b40675893 = messageFormatterFn((function(  ) {
  return function(d) { return "No existing data roles"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b187702700 = messageFormatterFn((function(  ) {
  return function(d) { return "Inherit synonyms from a published data role. In the list below, you can choose from all data roles you have permission to access."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b168950585 = messageFormatterFn((function(  ) {
  return function(d) { return "Apply Data Role"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2027668013 = messageFormatterFn((function(  ) {
  return function(d) { return ","; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1996121455 = messageFormatterFn((function(  ) {
  return function(d) { return ", "; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1365461479 = messageFormatterFn((function(  ) {
  return function(d) { return "Collapse"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a41215258 = messageFormatterFn((function(  ) {
  return function(d) { return "Expand"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2005619370 = messageFormatterFn((function(  ) {
  return function(d) { return "Inherit synonyms from a published data role"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a354086902 = messageFormatterFn((function(  ) {
  return function(d) { return "Data Role could not be applied."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1852162611 = messageFormatterFn((function(  ) {
  return function(d) { return "Synonyms could not be inherited because the data role does not exist"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a479618498 = messageFormatterFn((function(  ) {
  return function(d) { return "Synonyms could not be inherited because the user who associated the data role to this field can no longer view it"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a347549703 = messageFormatterFn((function(  ) {
  return function(d) { return "Apply Data Role..."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1794596129 = messageFormatterFn((function(  ) {
  return function(d) { return "Apply a Different Data Role..."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1849038214 = messageFormatterFn((function(  ) {
  return function(d) { return "Name"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a32733882 = messageFormatterFn((function(  ) {
  return function(d) { return "Owner"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a481521312 = messageFormatterFn((function(  ) {
  return function(d) { return "Project"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a709840417 = messageFormatterFn((function(  ) {
  return function(d) { return "Data Role"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1637700350 = messageFormatterFn((function(  ) {
  return function(d) { return "Inherited from geographic role"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b351679082 = messageFormatterFn((function(  ) {
  return function(d) { return "No data role applied"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1754264445 = messageFormatterFn((function(  ) {
  return function(d) { return "applied this data role. Their permissions are used to inherit synonyms."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a14335517 = messageFormatterFn((function(  ) {
  return function(d) { return "Failed to list all data roles."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2128751642 = messageFormatterFn((function(  ) {
  return function(d) { return "Retry"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b11733621 = messageFormatterFn((function(  ) {
  return function(d) { return "Make this field's synonyms inheritable by other data sources"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1849333830 = messageFormatterFn((function(  ) {
  return function(d) { return "Publish Data Role..."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1766293127 = messageFormatterFn((function(  ) {
  return function(d) { return "Remove Data Role"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1165687202 = messageFormatterFn((function(  ) {
  return function(d) { return "Failed to remove current data role."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1137135928 = messageFormatterFn((function(  ) {
  return function(d) { return "View Data Role"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1862988796 = messageFormatterFn((function(  ) {
  return function(d) { return "Boolean"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1052631169 = messageFormatterFn((function(  ) {
  return function(d) { return "Date-time"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1269385486 = messageFormatterFn((function(  ) {
  return function(d) { return "Date"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1125961170 = messageFormatterFn((function(  ) {
  return function(d) { return "Number (Whole)"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a392483102 = messageFormatterFn((function(  ) {
  return function(d) { return "Number"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1140560984 = messageFormatterFn((function(  ) {
  return function(d) { return "Spatial Data"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2113374539 = messageFormatterFn((function(  ) {
  return function(d) { return "String"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1061161886 = messageFormatterFn((function(  ) {
  return function(d) { return "Table"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a926655243 = messageFormatterFn((function(  ) {
  return function(d) { return "Dimensions"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a581116592 = messageFormatterFn((function(  ) {
  return function(d) { return "Could not save changes."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b376246326 = messageFormatterFn((function(  ) {
  return function(d) { return "Edit Field Details"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1697870004 = messageFormatterFn((function(  ) {
  return function(d) { return "Synonyms"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1306274044 = messageFormatterFn((function(  ) {
  return function(d) { return "Data Role"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1759342009 = messageFormatterFn((function(  ) {
  return function(d) { return "Type"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b519006871 = messageFormatterFn((function(  ) {
  return function(d) { return "Bin Field"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a287877302 = messageFormatterFn((function(  ) {
  return function(d) { return "Calculated Field"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b897090226 = messageFormatterFn((function(  ) {
  return function(d) { return "Column Field"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1690816253 = messageFormatterFn((function(  ) {
  return function(d) { return "Combined Field"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a687823871 = messageFormatterFn((function(  ) {
  return function(d) { return "Combined Set Field"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2117421153 = messageFormatterFn((function(  ) {
  return function(d) { return "Group Field"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a647212673 = messageFormatterFn((function(  ) {
  return function(d) { return "Hierarchical Field"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b357575676 = messageFormatterFn((function(  ) {
  return function(d) { return "Set Field"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a928312927 = messageFormatterFn((function(  ) {
  return function(d) { return "Field Details"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2070104393 = messageFormatterFn((function(  ) {
  return function(d) { return "Value"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1723499716 = messageFormatterFn((function(  ) {
  return function(d) { return "For immediate effect, click here to Reindex the Data Source"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b510823698 = messageFormatterFn((function(  ) {
  return function(d) { return "Value Indexing"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b868980012 = messageFormatterFn((function(  ) {
  return function(d) { return "Index field values"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1476269456 = messageFormatterFn((function(  ) {
  return function(d) { return "There was an error retrieving indexing settings for field"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1600944121 = messageFormatterFn((function(  ) {
  return function(d) { return "Ask Data couldn't index this field so some analytical features are disabled. Republish the data source to retry indexing."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b221918900 = messageFormatterFn((function(  ) {
  return function(d) { return "Values won't be searchable in Ask Data."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1016506861 = messageFormatterFn((function(  ) {
  return function(d) { return "Failed to index field."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2061094365 = messageFormatterFn((function(  ) {
  return function(d) { return "This field has no values."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b991797154 = messageFormatterFn((function(  ) {
  return function(d) { return "Ask Data couldn't index this field so some analytical features are disabled."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2024162055 = messageFormatterFn((function(  ) {
  return function(d) { return "String values won't be searchable in Ask Data."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1087277240 = messageFormatterFn((function(  ) {
  return function(d) { return "Field indexing disabled due to user filters."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1394061888 = messageFormatterFn((function(  ) {
  return function(d) { return "Ask Data couldn't index this field so some analytical features are disabled."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1118035739 = messageFormatterFn((function(  ) {
  return function(d) { return "Values won't be searchable in Ask Data."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1682347878 = messageFormatterFn((function(  ) {
  return function(d) { return "Failed to index field."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2084008367 = messageFormatterFn((function(  ) {
  return function(d) { return "Values won't be searchable in Ask Data."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a712690088 = messageFormatterFn((function(  ) {
  return function(d) { return "Failed to index field."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1281751069 = messageFormatterFn((function(  ) {
  return function(d) { return "Fix and republish data source to make this available in Ask Data."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1641897948 = messageFormatterFn((function(  ) {
  return function(d) { return "Formula is invalid."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1952072053 = messageFormatterFn((function(  ) {
  return function(d) { return "Ask Data doesn’t support indexing this field and won’t be able to recognize unquoted values at this time."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a971576964 = messageFormatterFn((function(  ) {
  return function(d) { return "Indexing isn't supported."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b877821412 = messageFormatterFn((function(  ) {
  return function(d) { return "Fix and republish data source to use this in Ask Data."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b606065093 = messageFormatterFn((function(  ) {
  return function(d) { return "Field names must be unique."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b513090757 = messageFormatterFn((function(  ) {
  return function(d) { return "Fields of type Latitude or Longitude aren’t available in Ask Data."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a60263089 = messageFormatterFn((function(  ) {
  return function(d) { return "This field is unsupported by Ask Data because it contains a constant formula that references parameters."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1060619074 = messageFormatterFn((function(  ) {
  return function(d) { return "Unsupported aggregation type. Fix and republish data source to make this available in Ask Data."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a793580197 = messageFormatterFn((function(  ) {
  return function(d) { return "This field’s name contains an invalid character and won’t be available in Ask Data."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2030009374 = messageFormatterFn((function(  ) {
  return function(d) { return "The field name can’t be a number, value, or date and won’t be available in Ask Data."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b308339178 = messageFormatterFn((function(  ) {
  return function(d) { return "This field’s name is longer than the 50 character limit. This field won’t be available in Ask Data."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b993050558 = messageFormatterFn((function(  ) {
  return function(d) { return "The field name is a reserved term and won’t be available in Ask Data."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a749902776 = messageFormatterFn((function(  ) {
  return function(d) { return "Field names must be at least 1 character long. This field won’t be available in Ask Data."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1935218087 = messageFormatterFn((function(  ) {
  return function(d) { return "Cluster fields aren't available in Ask Data."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b537386290 = messageFormatterFn((function(  ) {
  return function(d) { return "Combined fields aren't available in Ask Data."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1886249718 = messageFormatterFn((function(  ) {
  return function(d) { return "Combined set fields aren't available in Ask Data."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1992110430 = messageFormatterFn((function(  ) {
  return function(d) { return "Hierarchy fields aren't available in Ask Data."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1688113393 = messageFormatterFn((function(  ) {
  return function(d) { return "Set fields aren't available in Ask Data."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1775870985 = messageFormatterFn((function(  ) {
  return function(d) { return "Unknown field type. Fix and republish data source to make this available in Ask Data."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b198478212 = messageFormatterFn((function(  ) {
  return function(d) { return "Table calculations aren’t available in Ask Data."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2114590217 = messageFormatterFn((function(  ) {
  return function(d) { return "Revert to original name"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1915056602 = messageFormatterFn((function(  ) {
  return function(d) { return "Edit Field Name"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b534110968 = messageFormatterFn((function(  ) {
  return function(d) { return "Field names must be at least " + d.minLength + " character(s) long"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2005545670 = messageFormatterFn((function(  ) {
  return function(d) { return "Field names have a maximum length of " + d.maxLength + " characters"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a148901523 = messageFormatterFn((function(  ) {
  return function(d) { return "This name is already in use"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2007668611 = messageFormatterFn((function(  ) {
  return function(d) { return "Numbers, booleans, dates or null are not allowed"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1950872051 = messageFormatterFn((function(  ) {
  return function(d) { return "Encountered an error while trying to edit this name"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1568142134 = messageFormatterFn((function(  ) {
  return function(d) { return "Description"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1040293756 = messageFormatterFn((function(  ) {
  return function(d) { return "Details"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a619566114 = messageFormatterFn((function(  ) {
  return function(d) { return "Domain Values"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1095843010 = messageFormatterFn((function(  ) {
  return function(d) { return "Field Info"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a548193696 = messageFormatterFn((function(  ) {
  return function(d) { return "Formula"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b429479692 = messageFormatterFn((function(  ) {
  return function(d) { return "Synonyms"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a990196815 = messageFormatterFn((function(  ) {
  return function(d) { return "Value Distribution"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1765333779 = messageFormatterFn((function( plural, pluralFuncs ) {
  return function(d) { return d.rowCount + " " + plural(d.rowCounter, 0, pluralFuncs.en, { one: function() { return "row";}, other: function() { return "rows";} }); }
})(messageFormat.plural, {"en": Globalize("en").pluralGenerator()}), Globalize("en").pluralGenerator({}));
Globalize.a1109258637 = messageFormatterFn((function( plural, pluralFuncs ) {
  return function(d) { return d.valueCount + " " + plural(d.valueCounter, 0, pluralFuncs.en, { one: function() { return "unique value";}, other: function() { return "unique values";} }); }
})(messageFormat.plural, {"en": Globalize("en").pluralGenerator()}), Globalize("en").pluralGenerator({}));
Globalize.a1465657229 = messageFormatterFn((function(  ) {
  return function(d) { return "End"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b655644378 = messageFormatterFn((function(  ) {
  return function(d) { return "Start"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1923832793 = messageFormatterFn((function(  ) {
  return function(d) { return "Null"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2049386416 = messageFormatterFn((function(  ) {
  return function(d) { return "Average"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1344502035 = messageFormatterFn((function(  ) {
  return function(d) { return "Maximum"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1405605002 = messageFormatterFn((function(  ) {
  return function(d) { return "Min, Max"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b687585985 = messageFormatterFn((function(  ) {
  return function(d) { return "Minimum"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1562805369 = messageFormatterFn((function(  ) {
  return function(d) { return "The first " + d.maxFetch + " values are recognized in Ask Data."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1623464274 = messageFormatterFn((function(  ) {
  return function(d) { return d.role + " (" + d.dataType + ") "; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a962561399 = messageFormatterFn((function(  ) {
  return function(d) { return "Find Field"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1026279704 = messageFormatterFn((function(  ) {
  return function(d) { return "Apply"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1241971158 = messageFormatterFn((function(  ) {
  return function(d) { return "Cancel"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a817285257 = messageFormatterFn((function(  ) {
  return function(d) { return "Publish"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b701817868 = messageFormatterFn((function(  ) {
  return function(d) { return "Submit"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b285270079 = messageFormatterFn((function(  ) {
  return function(d) { return "A generated field that counts the number of rows."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b705499083 = messageFormatterFn((function(  ) {
  return function(d) { return "A generated field that counts the number of rows in the \"" + d.objectName + "\" table."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1532624323 = messageFormatterFn((function(  ) {
  return function(d) { return "A generated field that counts the number of rows in the table."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b981003069 = messageFormatterFn((function(  ) {
  return function(d) { return "Failed to index " + d.count + " field(s)"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1534761570 = messageFormatterFn((function(  ) {
  return function(d) { return "Republish it to finish indexing all fields, or try again."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1971150847 = messageFormatterFn((function(  ) {
  return function(d) { return "Failed to complete indexing after losing connection to the data source."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1908834813 = messageFormatterFn((function(  ) {
  return function(d) { return "String values won't be searchable in Ask Data."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1984541276 = messageFormatterFn((function(  ) {
  return function(d) { return "Indexing failed due to a system error."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2000930608 = messageFormatterFn((function(  ) {
  return function(d) { return "String values won't be searchable in Ask Data."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2001950641 = messageFormatterFn((function(  ) {
  return function(d) { return "Indexing has been disabled by the data source owner."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2019274323 = messageFormatterFn((function(  ) {
  return function(d) { return "String values won't be searchable in Ask Data."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b607519694 = messageFormatterFn((function(  ) {
  return function(d) { return "Indexing failed due to a system error."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b390443711 = messageFormatterFn((function(  ) {
  return function(d) { return "Republish with fewer fields so Ask Data can quickly retrieve string values."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1669299872 = messageFormatterFn((function(  ) {
  return function(d) { return "Data source has too many fields to index."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1822863571 = messageFormatterFn((function(  ) {
  return function(d) { return "Fix and republish data source to use this in Ask Data."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b678556788 = messageFormatterFn((function(  ) {
  return function(d) { return "Duplicate field name."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2033285784 = messageFormatterFn((function(  ) {
  return function(d) { return "Remove them from the data source to enable Ask Data to answer more questions and recognize field values."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1983802233 = messageFormatterFn((function(  ) {
  return function(d) { return "Indexing disabled due to user filters."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a183232298 = messageFormatterFn((function(  ) {
  return function(d) { return "Create an extract and republish to improve performance. String values won't be searchable in Ask Data."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1576042185 = messageFormatterFn((function(  ) {
  return function(d) { return "Data source is too slow to index."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a658154965 = messageFormatterFn((function(  ) {
  return function(d) { return "Ask Data can then answer more questions and recognize unquoted field values."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1784353740 = messageFormatterFn((function(  ) {
  return function(d) { return "Embed data source credentials to enable indexing so Ask Data can quickly retrieve string values."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a941317334 = messageFormatterFn((function(  ) {
  return function(d) { return "Create a lens to index values."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b778473611 = messageFormatterFn((function(  ) {
  return function(d) { return "Ask Data will index values for a datasource only if there is an associated lens."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1928096193 = messageFormatterFn((function(  ) {
  return function(d) { return "Indexing allows Ask Data to quickly retrieve field values."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b456308682 = messageFormatterFn((function(  ) {
  return function(d) { return "Last completed:"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1494108158 = messageFormatterFn((function(  ) {
  return function(d) { return "Data distributions last updated:"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b756339309 = messageFormatterFn((function(  ) {
  return function(d) { return "Updating data distributions " + d.count + "..."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2056770807 = messageFormatterFn((function(  ) {
  return function(d) { return "Indexing Data Source..."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1203608928 = messageFormatterFn((function(  ) {
  return function(d) { return "Indexing values..."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2017879100 = messageFormatterFn((function(  ) {
  return function(d) { return "Indexing " + d.fieldCountLoc; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1291612035 = messageFormatterFn((function(  ) {
  return function(d) { return "Waiting to retry."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1227511183 = messageFormatterFn((function(  ) {
  return function(d) { return "Index now"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1019050548 = messageFormatterFn((function(  ) {
  return function(d) { return "Update now"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1779977400 = messageFormatterFn((function(  ) {
  return function(d) { return "Index now"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b338609485 = messageFormatterFn((function(  ) {
  return function(d) { return "Refresh Data Source Fields"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b237283311 = messageFormatterFn((function(  ) {
  return function(d) { return "Retry indexing"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b949490748 = messageFormatterFn((function(  ) {
  return function(d) { return "Data Indexed"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b498804593 = messageFormatterFn((function(  ) {
  return function(d) { return "Try again"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b724607994 = messageFormatterFn((function( plural, pluralFuncs ) {
  return function(d) { return d.fieldCountLoc + " eligible " + plural(d.fieldCount, 0, pluralFuncs.en, { one: function() { return "field";}, other: function() { return "fields";} }); }
})(messageFormat.plural, {"en": Globalize("en").pluralGenerator()}), Globalize("en").pluralGenerator({}));
Globalize.b2000252548 = messageFormatterFn((function( plural, pluralFuncs ) {
  return function(d) { return d.fieldCountLoc + " unsupported " + plural(d.fieldCount, 0, pluralFuncs.en, { one: function() { return "field";}, other: function() { return "fields";} }); }
})(messageFormat.plural, {"en": Globalize("en").pluralGenerator()}), Globalize("en").pluralGenerator({}));
Globalize.b1938488831 = messageFormatterFn((function(  ) {
  return function(d) { return "Indexed:"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a978422912 = messageFormatterFn((function(  ) {
  return function(d) { return "Skipped:"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1662964244 = messageFormatterFn((function(  ) {
  return function(d) { return "Data"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a947789382 = messageFormatterFn((function(  ) {
  return function(d) { return "Data Content"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a623232638 = messageFormatterFn((function(  ) {
  return function(d) { return "Indexing Details"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b510713991 = messageFormatterFn((function(  ) {
  return function(d) { return "From data source"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1965332616 = messageFormatterFn((function(  ) {
  return function(d) { return "Revert to data source content"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1854080967 = messageFormatterFn((function(  ) {
  return function(d) { return "Field descriptions have a maximum length of " + d.maxLength + " characters"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1479563570 = messageFormatterFn((function(  ) {
  return function(d) { return "Encountered an error while trying to edit this description (error: " + d.errorCode + ")"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1173083595 = messageFormatterFn((function(  ) {
  return function(d) { return "Field name is too long. Rename field in data source or lens for a better Ask Data experience."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b513633418 = messageFormatterFn((function(  ) {
  return function(d) { return "Select Fields"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a79560278 = messageFormatterFn((function(  ) {
  return function(d) { return "Select the data source fields to be used in the lens."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1066977492 = messageFormatterFn((function(  ) {
  return function(d) { return "Renamed to [" + d.lens_display_name + "] in lens"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1076704413 = messageFormatterFn((function(  ) {
  return function(d) { return "Select All"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1356973324 = messageFormatterFn((function(  ) {
  return function(d) { return "Select None"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1621834551 = messageFormatterFn((function(  ) {
  return function(d) { return "[ " + d.field + " ] in data source"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1499711871 = messageFormatterFn((function(  ) {
  return function(d) { return "Lens fields successfully updated."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2062536698 = messageFormatterFn((function(  ) {
  return function(d) { return "Encountered an error while trying to update Lens fields."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a997767652 = messageFormatterFn((function(  ) {
  return function(d) { return "Encountered an error while checking the Lens edit permission."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1891411081 = messageFormatterFn((function(  ) {
  return function(d) { return "Name"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1693190484 = messageFormatterFn((function(  ) {
  return function(d) { return "No dimensions"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1354079384 = messageFormatterFn((function(  ) {
  return function(d) { return "No fields"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1894511941 = messageFormatterFn((function(  ) {
  return function(d) { return "No measures"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1957601813 = messageFormatterFn((function( plural, pluralFuncs ) {
  return function(d) { return d.formattedItemCount + " Field " + plural(d.itemCount, 0, pluralFuncs.en, { one: function() { return "Synonym";}, other: function() { return "Synonyms";} }); }
})(messageFormat.plural, {"en": Globalize("en").pluralGenerator()}), Globalize("en").pluralGenerator({}));
Globalize.b823420404 = messageFormatterFn((function( plural, pluralFuncs ) {
  return function(d) { return d.formattedItemCount + " Value " + plural(d.itemCount, 0, pluralFuncs.en, { one: function() { return "Synonym";}, other: function() { return "Synonyms";} }); }
})(messageFormat.plural, {"en": Globalize("en").pluralGenerator()}), Globalize("en").pluralGenerator({}));
Globalize.b1057828416 = messageFormatterFn((function(  ) {
  return function(d) { return "Data Role Contents"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b438472152 = messageFormatterFn((function(  ) {
  return function(d) { return "Data role published but could not be applied"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1243453520 = messageFormatterFn((function(  ) {
  return function(d) { return "Data role could not be published"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1795586901 = messageFormatterFn((function(  ) {
  return function(d) { return "Data role could not be published - Permission was denied"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b339792981 = messageFormatterFn((function(  ) {
  return function(d) { return "Data role published and applied"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1772326611 = messageFormatterFn((function(  ) {
  return function(d) { return "Publish data role to make this field's synonyms inheritable by other data sources."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1525055444 = messageFormatterFn((function(  ) {
  return function(d) { return "The data role will be created with default permissions for the project."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1002654824 = messageFormatterFn((function(  ) {
  return function(d) { return "Publish Data Role"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1947870214 = messageFormatterFn((function(  ) {
  return function(d) { return "Only values with synonyms will be copied from this field to the data role."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1876821868 = messageFormatterFn((function(  ) {
  return function(d) { return "A data role with this name already exists. Publishing will overwrite the existing data role."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1335079979 = messageFormatterFn((function(  ) {
  return function(d) { return "Table names work best as a plural noun that describes what the table rows collectively represent. For example, \"Orders\" or \"Customers\"."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b430022962 = messageFormatterFn((function(  ) {
  return function(d) { return "Search"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1953873633 = messageFormatterFn((function(  ) {
  return function(d) { return "Contains an invalid character."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1383754140 = messageFormatterFn((function(  ) {
  return function(d) { return "Synonyms must not be numbers, booleans, dates or null."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2068718337 = messageFormatterFn((function(  ) {
  return function(d) { return "Synonyms must be less than " + d.max + " characters."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2118104953 = messageFormatterFn((function(  ) {
  return function(d) { return "Synonyms can't contain reserved terms."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b502332396 = messageFormatterFn((function(  ) {
  return function(d) { return "The maximum number of synonyms is 25"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1935166221 = messageFormatterFn((function(  ) {
  return function(d) { return "Synonyms must be at least 1 character."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b596157988 = messageFormatterFn((function(  ) {
  return function(d) { return "Synonyms for  \"" + d.value + "\""; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1859594973 = messageFormatterFn((function(  ) {
  return function(d) { return "Tables"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1341924804 = messageFormatterFn((function(  ) {
  return function(d) { return "Synonym could not be published."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1242544069 = messageFormatterFn((function(  ) {
  return function(d) { return "Synonym could not be published."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b104083701 = messageFormatterFn((function(  ) {
  return function(d) { return "Search"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a296950726 = messageFormatterFn((function(  ) {
  return function(d) { return "This field already has a filter. Edit or remove the existing one."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1067487979 = messageFormatterFn((function(  ) {
  return function(d) { return "Can't use this field since there are no more aggregation types available."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1899935836 = messageFormatterFn((function(  ) {
  return function(d) { return "Ask Data"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1031269282 = messageFormatterFn((function(  ) {
  return function(d) { return d.baseTitle + " - Ask Data"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1766332048 = messageFormatterFn((function(  ) {
  return function(d) { return "/"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1361181569 = messageFormatterFn((function(  ) {
  return function(d) { return "Basic Data Analysis"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2206573 = messageFormatterFn((function(  ) {
  return function(d) { return "Simple Calculations"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b679821511 = messageFormatterFn((function(  ) {
  return function(d) { return "Date and Time"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1050225138 = messageFormatterFn((function(  ) {
  return function(d) { return "Filters"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b354914539 = messageFormatterFn((function(  ) {
  return function(d) { return "Viz Type"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1762385778 = messageFormatterFn((function(  ) {
  return function(d) { return "All"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1153447684 = messageFormatterFn((function(  ) {
  return function(d) { return "Contains"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1146013955 = messageFormatterFn((function(  ) {
  return function(d) { return "Ends with"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b587916893 = messageFormatterFn((function(  ) {
  return function(d) { return "Does not contain"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b595350622 = messageFormatterFn((function(  ) {
  return function(d) { return "Does not end with"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b428790789 = messageFormatterFn((function(  ) {
  return function(d) { return "Does not start with"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b872915687 = messageFormatterFn((function(  ) {
  return function(d) { return "Specific Values"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2014677732 = messageFormatterFn((function(  ) {
  return function(d) { return "Starts with"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b410206420 = messageFormatterFn((function(  ) {
  return function(d) { return "Categorical filter tabs"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b977865729 = messageFormatterFn((function(  ) {
  return function(d) { return "Wildcard"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1795671093 = messageFormatterFn((function(  ) {
  return function(d) { return "Copy"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b148202613 = messageFormatterFn((function(  ) {
  return function(d) { return "Copy link"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b128730966 = messageFormatterFn((function(  ) {
  return function(d) { return "Edit"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1733216739 = messageFormatterFn((function(  ) {
  return function(d) { return "This group is empty and hidden from others until you " + d.addAQuestion + "."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1742652515 = messageFormatterFn((function(  ) {
  return function(d) { return "Edit Title"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a85664815 = messageFormatterFn((function(  ) {
  return function(d) { return "Edit Visualization"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1705363616 = messageFormatterFn((function(  ) {
  return function(d) { return "This section is empty and hidden from others."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1579368810 = messageFormatterFn((function(  ) {
  return function(d) { return "pin icon in the toolbar."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2033287684 = messageFormatterFn((function(  ) {
  return function(d) { return "To add or replace links in this list, go to a visualization and click the"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1891282163 = messageFormatterFn((function(  ) {
  return function(d) { return "Recommended Visualizations"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1324678491 = messageFormatterFn((function(  ) {
  return function(d) { return "Section Name"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a955172586 = messageFormatterFn((function(  ) {
  return function(d) { return "We couldn't save your changes"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1355897810 = messageFormatterFn((function(  ) {
  return function(d) { return "Try Again"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1604683719 = messageFormatterFn((function(  ) {
  return function(d) { return "AM"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1605130584 = messageFormatterFn((function(  ) {
  return function(d) { return "PM"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1859446206 = messageFormatterFn((function(  ) {
  return function(d) { return "Day"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1680855660 = messageFormatterFn((function(  ) {
  return function(d) { return "Hour"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1495705468 = messageFormatterFn((function(  ) {
  return function(d) { return "Minute"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b430706330 = messageFormatterFn((function(  ) {
  return function(d) { return "Month"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1356650542 = messageFormatterFn((function(  ) {
  return function(d) { return "Quarter"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a795533028 = messageFormatterFn((function(  ) {
  return function(d) { return "Second"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1261136988 = messageFormatterFn((function(  ) {
  return function(d) { return "Week"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b348234639 = messageFormatterFn((function(  ) {
  return function(d) { return "Week " + d.weekNumber; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1203991123 = messageFormatterFn((function(  ) {
  return function(d) { return "Year"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a663590393 = messageFormatterFn((function(  ) {
  return function(d) { return "day"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b776132419 = messageFormatterFn((function(  ) {
  return function(d) { return "hour"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a359935341 = messageFormatterFn((function(  ) {
  return function(d) { return "minute"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1845910365 = messageFormatterFn((function(  ) {
  return function(d) { return "month"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1213937963 = messageFormatterFn((function(  ) {
  return function(d) { return "Month and Day"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1917204094 = messageFormatterFn((function(  ) {
  return function(d) { return "Month, Day, and Time"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a491350404 = messageFormatterFn((function(  ) {
  return function(d) { return "April"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2109008357 = messageFormatterFn((function(  ) {
  return function(d) { return "August"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1150352259 = messageFormatterFn((function(  ) {
  return function(d) { return "December"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b110072646 = messageFormatterFn((function(  ) {
  return function(d) { return "February"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1362518440 = messageFormatterFn((function(  ) {
  return function(d) { return "January"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b830414288 = messageFormatterFn((function(  ) {
  return function(d) { return "July"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b830373926 = messageFormatterFn((function(  ) {
  return function(d) { return "June"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2121840129 = messageFormatterFn((function(  ) {
  return function(d) { return "March"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1776512355 = messageFormatterFn((function(  ) {
  return function(d) { return "May"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b961537462 = messageFormatterFn((function(  ) {
  return function(d) { return "November"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a964643486 = messageFormatterFn((function(  ) {
  return function(d) { return "October"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1961341839 = messageFormatterFn((function(  ) {
  return function(d) { return "September"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a333639689 = messageFormatterFn((function(  ) {
  return function(d) { return "quarter"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1643793459 = messageFormatterFn((function(  ) {
  return function(d) { return "second"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b356413747 = messageFormatterFn((function(  ) {
  return function(d) { return "week"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b299267882 = messageFormatterFn((function(  ) {
  return function(d) { return "year"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a171242461 = messageFormatterFn((function(  ) {
  return function(d) { return "Date Detail"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1827882127 = messageFormatterFn((function(  ) {
  return function(d) { return "Invalid year"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b627857349 = messageFormatterFn((function(  ) {
  return function(d) { return "Date Value"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2069691612 = messageFormatterFn((function(  ) {
  return function(d) { return "End Date"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b692964614 = messageFormatterFn((function(  ) {
  return function(d) { return "Following " + d.timeUnit; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a31396502 = messageFormatterFn((function(  ) {
  return function(d) { return "Includes this " + d.timeUnit; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a319885445 = messageFormatterFn((function(  ) {
  return function(d) { return "Includes today"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a443342787 = messageFormatterFn((function(  ) {
  return function(d) { return d.dateValue + " (No data)"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1984969298 = messageFormatterFn((function(  ) {
  return function(d) { return "Day Numbers"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1522202217 = messageFormatterFn((function(  ) {
  return function(d) { return "Months"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1512245227 = messageFormatterFn((function(  ) {
  return function(d) { return "Quarters"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a780732138 = messageFormatterFn((function(  ) {
  return function(d) { return "Week Numbers"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1747490193 = messageFormatterFn((function(  ) {
  return function(d) { return "Weekdays"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1852011196 = messageFormatterFn((function(  ) {
  return function(d) { return "days"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1705174106 = messageFormatterFn((function(  ) {
  return function(d) { return "hours"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1188656586 = messageFormatterFn((function(  ) {
  return function(d) { return "minutes"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1457001312 = messageFormatterFn((function(  ) {
  return function(d) { return "months"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a254194508 = messageFormatterFn((function(  ) {
  return function(d) { return "quarters"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2124892778 = messageFormatterFn((function(  ) {
  return function(d) { return "weeks"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2112928653 = messageFormatterFn((function(  ) {
  return function(d) { return "years"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b892964268 = messageFormatterFn((function(  ) {
  return function(d) { return "Previous " + d.timeUnit; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b603417293 = messageFormatterFn((function(  ) {
  return function(d) { return "Last"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b542315952 = messageFormatterFn((function(  ) {
  return function(d) { return "Next"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b184779016 = messageFormatterFn((function(  ) {
  return function(d) { return "Relative Dates"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a268761663 = messageFormatterFn((function(  ) {
  return function(d) { return d.firstPeriod + " to " + d.lastPeriod; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1507450607 = messageFormatterFn((function(  ) {
  return function(d) { return "Day 8"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a465747250 = messageFormatterFn((function(  ) {
  return function(d) { return "May"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1931372320 = messageFormatterFn((function(  ) {
  return function(d) { return "May 8"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b187130355 = messageFormatterFn((function(  ) {
  return function(d) { return "May 8, 10:32pm"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a280931600 = messageFormatterFn((function(  ) {
  return function(d) { return "Q1"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a285826934 = messageFormatterFn((function(  ) {
  return function(d) { return "Q2"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b622885240 = messageFormatterFn((function(  ) {
  return function(d) { return "Q3"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b137242030 = messageFormatterFn((function(  ) {
  return function(d) { return "Q4"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a76996090 = messageFormatterFn((function(  ) {
  return function(d) { return "Friday"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a812324289 = messageFormatterFn((function(  ) {
  return function(d) { return "Week 24"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1595873377 = messageFormatterFn((function(  ) {
  return function(d) { return "2020"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a354269601 = messageFormatterFn((function(  ) {
  return function(d) { return "May 8, 2020"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1775142379 = messageFormatterFn((function(  ) {
  return function(d) { return "May 8, 2020 at 10 AM"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1910929211 = messageFormatterFn((function(  ) {
  return function(d) { return "May 8, 2020 at 10:32 AM"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a941372677 = messageFormatterFn((function(  ) {
  return function(d) { return "May 2020"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1343684687 = messageFormatterFn((function(  ) {
  return function(d) { return "Q1 2020"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a380309285 = messageFormatterFn((function(  ) {
  return function(d) { return "May 8, 2020 at 10:32:12 AM"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1355423707 = messageFormatterFn((function(  ) {
  return function(d) { return "Week 5, 2020"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1864541778 = messageFormatterFn((function(  ) {
  return function(d) { return "Specific Dates"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1324141931 = messageFormatterFn((function(  ) {
  return function(d) { return "Start Date"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1759698087 = messageFormatterFn((function(  ) {
  return function(d) { return "Range of Dates"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b655422354 = messageFormatterFn((function(  ) {
  return function(d) { return "MM/DD/YYYY"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1129252229 = messageFormatterFn((function(  ) {
  return function(d) { return "Time"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b352323369 = messageFormatterFn((function(  ) {
  return function(d) { return "Range Type"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1539145408 = messageFormatterFn((function(  ) {
  return function(d) { return "Between"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b327862907 = messageFormatterFn((function(  ) {
  return function(d) { return "Ending"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a680343966 = messageFormatterFn((function(  ) {
  return function(d) { return "Starting"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a772147897 = messageFormatterFn((function(  ) {
  return function(d) { return "Date filter tabs"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1210466227 = messageFormatterFn((function(  ) {
  return function(d) { return "This " + d.timeUnit; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1431018180 = messageFormatterFn((function(  ) {
  return function(d) { return "Today"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1697517440 = messageFormatterFn((function(  ) {
  return function(d) { return "Tomorrow"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b307173571 = messageFormatterFn((function(  ) {
  return function(d) { return "Yesterday"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a809470561 = messageFormatterFn((function(  ) {
  return function(d) { return "Description"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2006107283 = messageFormatterFn((function(  ) {
  return function(d) { return "Deselect All"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a441850909 = messageFormatterFn((function(  ) {
  return function(d) { return "Field"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a812552396 = messageFormatterFn((function(  ) {
  return function(d) { return "Fields"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b410421642 = messageFormatterFn((function(  ) {
  return function(d) { return "Time Period"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1098369145 = messageFormatterFn((function(  ) {
  return function(d) { return "Edit workbook"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b829446534 = messageFormatterFn((function(  ) {
  return function(d) { return "Ask Data embed code has been copied to your clipboard."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1609538196 = messageFormatterFn((function(  ) {
  return function(d) { return "Embed Ask Data for this lens in your HTML page. No sheets or vizzes will get embeded."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a455383682 = messageFormatterFn((function(  ) {
  return function(d) { return "Copy and paste this into any HTML page."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1288917692 = messageFormatterFn((function(  ) {
  return function(d) { return "Embed Code for Ask Data"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1413981775 = messageFormatterFn((function(  ) {
  return function(d) { return "Aggregation"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1942212907 = messageFormatterFn((function(  ) {
  return function(d) { return "Boolean"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a994754538 = messageFormatterFn((function(  ) {
  return function(d) { return "Calculation"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a335496701 = messageFormatterFn((function(  ) {
  return function(d) { return "Filter"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b185414840 = messageFormatterFn((function(  ) {
  return function(d) { return "Filter Operator"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1433795252 = messageFormatterFn((function(  ) {
  return function(d) { return "Group"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a60852728 = messageFormatterFn((function(  ) {
  return function(d) { return "Limit"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b409642665 = messageFormatterFn((function(  ) {
  return function(d) { return "Sort"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1040994118 = messageFormatterFn((function(  ) {
  return function(d) { return "Table Calculation"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a755506 = messageFormatterFn((function(  ) {
  return function(d) { return "Viz Type"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a66865327 = messageFormatterFn((function(  ) {
  return function(d) { return "Filter"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1001068052 = messageFormatterFn((function(  ) {
  return function(d) { return "Date Filter"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1852015925 = messageFormatterFn((function(  ) {
  return function(d) { return "Limit Filter"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2108170905 = messageFormatterFn((function(  ) {
  return function(d) { return "Numeric Filter"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1958358225 = messageFormatterFn((function(  ) {
  return function(d) { return "String Filter"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1965589554 = messageFormatterFn((function(  ) {
  return function(d) { return "No results"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2028145013 = messageFormatterFn((function(  ) {
  return function(d) { return "No fields"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a595547477 = messageFormatterFn((function(  ) {
  return function(d) { return "No filters"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b777733095 = messageFormatterFn((function(  ) {
  return function(d) { return "Your search can't be applied to the visualization"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b941976666 = messageFormatterFn((function(  ) {
  return function(d) { return "Your search isn't understood"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1034737754 = messageFormatterFn((function(  ) {
  return function(d) { return "Your search can't be applied to the visualization"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b413973629 = messageFormatterFn((function(  ) {
  return function(d) { return "Your search isn't understood"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a288983175 = messageFormatterFn((function(  ) {
  return function(d) { return "Submit"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b349643234 = messageFormatterFn((function(  ) {
  return function(d) { return "Ask Data encountered an error and couldn’t process your search"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b872858099 = messageFormatterFn((function(  ) {
  return function(d) { return "Can’t process your request because it contains a hidden field. Click \"Clear All\" to start over."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b19549388 = messageFormatterFn((function(  ) {
  return function(d) { return "Not used in interpretation"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a485736704 = messageFormatterFn((function(  ) {
  return function(d) { return "Unspecified"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a976812433 = messageFormatterFn((function(  ) {
  return function(d) { return "contains:"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1104400567 = messageFormatterFn((function(  ) {
  return function(d) { return "in"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a304276696 = messageFormatterFn((function(  ) {
  return function(d) { return "Exclude"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1779617938 = messageFormatterFn((function(  ) {
  return function(d) { return "Help Improve Ask Data"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a81024025 = messageFormatterFn((function(  ) {
  return function(d) { return "Feedback"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b408285407 = messageFormatterFn((function(  ) {
  return function(d) { return "Build Version Info"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a180991980 = messageFormatterFn((function(  ) {
  return function(d) { return "Comments"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1489455701 = messageFormatterFn((function(  ) {
  return function(d) { return "Company (optional)"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b89277053 = messageFormatterFn((function(  ) {
  return function(d) { return "Send feedback directly to Tableau's Ask Data team. We'll see the last question you asked using the feature, comments you add here, and your basic system setup."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a928681249 = messageFormatterFn((function(  ) {
  return function(d) { return "Feedback Type"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2073489949 = messageFormatterFn((function(  ) {
  return function(d) { return "Submit Feedback"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1441629835 = messageFormatterFn((function(  ) {
  return function(d) { return "Name"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b799603199 = messageFormatterFn((function(  ) {
  return function(d) { return "How can we help you?"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a628644567 = messageFormatterFn((function(  ) {
  return function(d) { return "What do you like about this experience?"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1641463674 = messageFormatterFn((function(  ) {
  return function(d) { return "Explain what's happening and walk us through the problem."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2062365695 = messageFormatterFn((function(  ) {
  return function(d) { return "Attempted to connect to " + d.URL; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a241244148 = messageFormatterFn((function(  ) {
  return function(d) { return "Feedback connection issue. Please contact your Tableau Server administrator or IT department to diagnose."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b752217350 = messageFormatterFn((function(  ) {
  return function(d) { return "Unable to submit feedback."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b918400707 = messageFormatterFn((function(  ) {
  return function(d) { return "Thanks for your feedback!"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b265514095 = messageFormatterFn((function(  ) {
  return function(d) { return "Submitting, Please Wait"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1600887495 = messageFormatterFn((function(  ) {
  return function(d) { return " was created to track your input."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1316827782 = messageFormatterFn((function(  ) {
  return function(d) { return "Feature Request"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b62152271 = messageFormatterFn((function(  ) {
  return function(d) { return "Praise"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b613113037 = messageFormatterFn((function(  ) {
  return function(d) { return "Problem"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1158513739 = messageFormatterFn((function(  ) {
  return function(d) { return "If Ask Data isn’t producing the results you expect, email your question to the author of this lens. They will see your email address and the message you provide below."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1921429885 = messageFormatterFn((function(  ) {
  return function(d) { return "If Ask Data isn’t producing the results you expect, email your question to the author of this lens. They will see your message but not your name or contact info."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2056487185 = messageFormatterFn((function(  ) {
  return function(d) { return "Message"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2041197407 = messageFormatterFn((function(  ) {
  return function(d) { return "Send me a copy"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1699704603 = messageFormatterFn((function(  ) {
  return function(d) { return "Send Email"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a204390844 = messageFormatterFn((function(  ) {
  return function(d) { return "Unable to send your message due to an unexpected error."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1565066324 = messageFormatterFn((function(  ) {
  return function(d) { return "This request cannot be completed due to missing email information. Please ensure that both your and the lens author's email information is available before trying again."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a288297787 = messageFormatterFn((function(  ) {
  return function(d) { return "Your message has been sent!"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a691094099 = messageFormatterFn((function(  ) {
  return function(d) { return "Sending message, please wait..."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1519847023 = messageFormatterFn((function(  ) {
  return function(d) { return "Contact the Lens Author"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b804089992 = messageFormatterFn((function(  ) {
  return function(d) { return "All Types"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a302909488 = messageFormatterFn((function(  ) {
  return function(d) { return "Boolean"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2117016843 = messageFormatterFn((function(  ) {
  return function(d) { return "Boolean Field"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1763646428 = messageFormatterFn((function(  ) {
  return function(d) { return "Currency Field"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1362732897 = messageFormatterFn((function(  ) {
  return function(d) { return "Currency"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2077423193 = messageFormatterFn((function(  ) {
  return function(d) { return "Date Field"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2071043170 = messageFormatterFn((function(  ) {
  return function(d) { return "Date"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1682725942 = messageFormatterFn((function(  ) {
  return function(d) { return "Geographic"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2132288236 = messageFormatterFn((function(  ) {
  return function(d) { return "Number Field"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b437891737 = messageFormatterFn((function(  ) {
  return function(d) { return "Number"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1682479419 = messageFormatterFn((function(  ) {
  return function(d) { return "Percentage Field"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1023765238 = messageFormatterFn((function(  ) {
  return function(d) { return "Percentage"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a490848739 = messageFormatterFn((function(  ) {
  return function(d) { return "Booleans"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1293973391 = messageFormatterFn((function(  ) {
  return function(d) { return "Dates"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b660717918 = messageFormatterFn((function(  ) {
  return function(d) { return "Datetimes"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b611056344 = messageFormatterFn((function(  ) {
  return function(d) { return "Locations"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a122360148 = messageFormatterFn((function(  ) {
  return function(d) { return "Numbers"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b27452532 = messageFormatterFn((function(  ) {
  return function(d) { return "Strings"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b106624717 = messageFormatterFn((function(  ) {
  return function(d) { return "Table Rows"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b245914844 = messageFormatterFn((function(  ) {
  return function(d) { return "String Field"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1250687377 = messageFormatterFn((function(  ) {
  return function(d) { return "Table Row"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a36039171 = messageFormatterFn((function(  ) {
  return function(d) { return "Text"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a176335527 = messageFormatterFn((function(  ) {
  return function(d) { return "selected"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1907883453 = messageFormatterFn((function(  ) {
  return function(d) { return "At Least"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b454245627 = messageFormatterFn((function(  ) {
  return function(d) { return "At Most"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1387082575 = messageFormatterFn((function(  ) {
  return function(d) { return "Between"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a441275406 = messageFormatterFn((function(  ) {
  return function(d) { return "Accept"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1830340352 = messageFormatterFn((function(  ) {
  return function(d) { return "Cancel"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1837912316 = messageFormatterFn((function(  ) {
  return function(d) { return "Replace"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a219528707 = messageFormatterFn((function(  ) {
  return function(d) { return "Save"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a707884869 = messageFormatterFn((function(  ) {
  return function(d) { return "Contact the Lens Author"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a111068836 = messageFormatterFn((function(  ) {
  return function(d) { return "Answer your data questions as quickly as you can think of them."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1614210978 = messageFormatterFn((function(  ) {
  return function(d) { return "Customize this lens to give your users the best Ask Data experience."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a427947221 = messageFormatterFn((function(  ) {
  return function(d) { return "For Lens Authors"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1367285786 = messageFormatterFn((function(  ) {
  return function(d) { return "Hover over individual fields and click the pencil icon."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b337734014 = messageFormatterFn((function(  ) {
  return function(d) { return "Add synonyms for field names and values"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b570428731 = messageFormatterFn((function(  ) {
  return function(d) { return "To link to a visualization, while viewing it, click the pin icon in the toolbar."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1587479391 = messageFormatterFn((function(  ) {
  return function(d) { return "Customize the Recommended Visualizations list"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2095880212 = messageFormatterFn((function(  ) {
  return function(d) { return "Click the pencil icon at the top of the Fields panel."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a46893360 = messageFormatterFn((function(  ) {
  return function(d) { return "Hide unnecessary fields"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1543993094 = messageFormatterFn((function(  ) {
  return function(d) { return "Optimize for Ask Data"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1003725557 = messageFormatterFn((function(  ) {
  return function(d) { return "Do More"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1021446213 = messageFormatterFn((function(  ) {
  return function(d) { return "Due to a system error, we couldn't change the setting. Try again."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1157844213 = messageFormatterFn((function(  ) {
  return function(d) { return "Show Contact Form"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1397583550 = messageFormatterFn((function(  ) {
  return function(d) { return "Hide Contact Form"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1486207506 = messageFormatterFn((function(  ) {
  return function(d) { return "Give Product Feedback"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b84287384 = messageFormatterFn((function(  ) {
  return function(d) { return "Ask Data tips"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1783583167 = messageFormatterFn((function(  ) {
  return function(d) { return "Online Help"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b867014326 = messageFormatterFn((function(  ) {
  return function(d) { return "Setup Guide"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1480387939 = messageFormatterFn((function(  ) {
  return function(d) { return "Take a Tour"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1978201283 = messageFormatterFn((function(  ) {
  return function(d) { return "Ask Data"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1858564251 = messageFormatterFn((function(  ) {
  return function(d) { return "To see tips again, click here."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b919385040 = messageFormatterFn((function(  ) {
  return function(d) { return "Usage Analytics"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b911724613 = messageFormatterFn((function(  ) {
  return function(d) { return "Learn more"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2076662630 = messageFormatterFn((function(  ) {
  return function(d) { return "Connected Ask Data Lenses (" + d.numberOfLenses + ")"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1599420147 = messageFormatterFn((function(  ) {
  return function(d) { return "An error has occured."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1855216501 = messageFormatterFn((function(  ) {
  return function(d) { return "Failed to get the connected lenses of the datasource"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a762639649 = messageFormatterFn((function(  ) {
  return function(d) { return "Create a Lens to Use Ask Data"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1705927900 = messageFormatterFn((function(  ) {
  return function(d) { return "With a lens, you customize Ask Data for specific users by selecting the subset of data fields that are relevant to them, and adding synonyms they commonly use for field names and values."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2109842921 = messageFormatterFn((function(  ) {
  return function(d) { return "Create New Lens"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1595863483 = messageFormatterFn((function(  ) {
  return function(d) { return "New Ask Data Lens"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1667841108 = messageFormatterFn((function(  ) {
  return function(d) { return "An Ask Data Lens stores settings such as field choice, synonyms, and recommended visualizations."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1374409122 = messageFormatterFn((function(  ) {
  return function(d) { return "Lens Details"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1663792530 = messageFormatterFn((function(  ) {
  return function(d) { return "Location"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1866978024 = messageFormatterFn((function(  ) {
  return function(d) { return "Publish Lens"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1501417644 = messageFormatterFn((function(  ) {
  return function(d) { return "The lens name is already in use. Please try a different name."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1067716062 = messageFormatterFn((function(  ) {
  return function(d) { return "An unexpected error occurred while attempting to publish the lens “" + d.LENS_NAME + "”."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b77186834 = messageFormatterFn((function(  ) {
  return function(d) { return "Created lens “" + d.LENS_NAME + "”"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1919588789 = messageFormatterFn((function(  ) {
  return function(d) { return "Go to Lens"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b891954676 = messageFormatterFn((function(  ) {
  return function(d) { return "A lens with this name already exists. Please try a different name"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1123141033 = messageFormatterFn((function(  ) {
  return function(d) { return "Limit"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1431422085 = messageFormatterFn((function(  ) {
  return function(d) { return "by"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b389790957 = messageFormatterFn((function(  ) {
  return function(d) { return "Bottom"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a368700887 = messageFormatterFn((function(  ) {
  return function(d) { return "Top"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b248259495 = messageFormatterFn((function(  ) {
  return function(d) { return "Link has been copied to clipboard."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b789102488 = messageFormatterFn((function(  ) {
  return function(d) { return "To share the current sheet, send this link to people who can access this data source."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b54499480 = messageFormatterFn((function(  ) {
  return function(d) { return "Share Ask Data"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a527005327 = messageFormatterFn((function(  ) {
  return function(d) { return "en_US"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b394891931 = messageFormatterFn((function(  ) {
  return function(d) { return "Delete"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1092539973 = messageFormatterFn((function(  ) {
  return function(d) { return "Duplicate"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a778003354 = messageFormatterFn((function(  ) {
  return function(d) { return "New Sheet"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a468978178 = messageFormatterFn((function(  ) {
  return function(d) { return "Name"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1206026555 = messageFormatterFn((function(  ) {
  return function(d) { return "Search fields or values to create a visualization"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a567737639 = messageFormatterFn((function(  ) {
  return function(d) { return "Search fields or values to create a visualization (English only)"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2054095610 = messageFormatterFn((function(  ) {
  return function(d) { return "Search to add fields or filters to the visualization"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a981141429 = messageFormatterFn((function(  ) {
  return function(d) { return "Ask Data took too long to respond. If the issue persists, contact your administrator or Tableau support."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1901127648 = messageFormatterFn((function(  ) {
  return function(d) { return "Ask Data doesn't understand. Keep typing or choose a suggestion below."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b125797485 = messageFormatterFn((function(  ) {
  return function(d) { return "Ask Data encountered an error. If the issue persists, contact your administrator or Tableau support."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a67926031 = messageFormatterFn((function(  ) {
  return function(d) { return "To enable it, contact your Tableau administrator."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a743580833 = messageFormatterFn((function(  ) {
  return function(d) { return "Ask Data is disabled for this site"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b716430938 = messageFormatterFn((function(  ) {
  return function(d) { return "No data matches the filters for your question. "; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b938872436 = messageFormatterFn((function(  ) {
  return function(d) { return "No records"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a175659444 = messageFormatterFn((function(  ) {
  return function(d) { return "Got it"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b692912381 = messageFormatterFn((function(  ) {
  return function(d) { return "Page unavailable"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b887988095 = messageFormatterFn((function(  ) {
  return function(d) { return "Add Field"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1549930009 = messageFormatterFn((function(  ) {
  return function(d) { return "Add Filter"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a173920147 = messageFormatterFn((function(  ) {
  return function(d) { return "Advanced"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1072781949 = messageFormatterFn((function(  ) {
  return function(d) { return "Count"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a420401248 = messageFormatterFn((function(  ) {
  return function(d) { return "The number of rows"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1372669902 = messageFormatterFn((function(  ) {
  return function(d) { return "Distinct Count"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b497984981 = messageFormatterFn((function(  ) {
  return function(d) { return "The number of unique values"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a80358069 = messageFormatterFn((function(  ) {
  return function(d) { return "Earliest"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b985523566 = messageFormatterFn((function(  ) {
  return function(d) { return "The first date"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a31878569 = messageFormatterFn((function(  ) {
  return function(d) { return "Latest"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a351113094 = messageFormatterFn((function(  ) {
  return function(d) { return "The last date"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1455095788 = messageFormatterFn((function(  ) {
  return function(d) { return "Maximum"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a163895665 = messageFormatterFn((function(  ) {
  return function(d) { return "The largest value"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b682617214 = messageFormatterFn((function(  ) {
  return function(d) { return "Minimum"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1043053279 = messageFormatterFn((function(  ) {
  return function(d) { return "The smallest value"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1093129796 = messageFormatterFn((function(  ) {
  return function(d) { return "Change Field"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a680611852 = messageFormatterFn((function(  ) {
  return function(d) { return "Change Filter"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1455146142 = messageFormatterFn((function(  ) {
  return function(d) { return "What field do you want to filter?"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b615635679 = messageFormatterFn((function(  ) {
  return function(d) { return "How do you want to group these dates?"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b613443170 = messageFormatterFn((function(  ) {
  return function(d) { return "How do you want to use this field?"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1343038080 = messageFormatterFn((function(  ) {
  return function(d) { return "New Field"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1107595320 = messageFormatterFn((function(  ) {
  return function(d) { return "New Filter"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a111188982 = messageFormatterFn((function(  ) {
  return function(d) { return "Other"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1134846955 = messageFormatterFn((function(  ) {
  return function(d) { return "Search Fields"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1082994512 = messageFormatterFn((function(  ) {
  return function(d) { return "Select aggregation method"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1810537745 = messageFormatterFn((function(  ) {
  return function(d) { return "Unable to show values"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1584926183 = messageFormatterFn((function(  ) {
  return function(d) { return "The field hasn't been indexed so its values aren't searchable in Ask Data"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2020273838 = messageFormatterFn((function(  ) {
  return function(d) { return "What field do you want to visualize?"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1135152950 = messageFormatterFn((function(  ) {
  return function(d) { return "Suggestions"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1404333819 = messageFormatterFn((function(  ) {
  return function(d) { return "Suggestions are unavailable at this time."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2018711760 = messageFormatterFn((function(  ) {
  return function(d) { return "Quantitative filter tabs"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a148049902 = messageFormatterFn((function(  ) {
  return function(d) { return "Some linked visualizations are invalid. Go to a new visualization, click the pin icon in the toolbar, and then select Replace Recommendation."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1861008660 = messageFormatterFn((function(  ) {
  return function(d) { return "Pin to Recommended Visualizations..."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a228045873 = messageFormatterFn((function(  ) {
  return function(d) { return "Invalid visualization.  To replace, click the pin icon and select Replace Recommendation."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b425130299 = messageFormatterFn((function(  ) {
  return function(d) { return "Name must be unique"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b260996471 = messageFormatterFn((function(  ) {
  return function(d) { return "Name is required"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1079278945 = messageFormatterFn((function(  ) {
  return function(d) { return "Linked Visualization"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1656516481 = messageFormatterFn((function(  ) {
  return function(d) { return "Name"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2066181832 = messageFormatterFn((function(  ) {
  return function(d) { return "Section"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1706499734 = messageFormatterFn((function(  ) {
  return function(d) { return "Recommended Visualization"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1240112633 = messageFormatterFn((function(  ) {
  return function(d) { return "Saved to Recommended Visualizations"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1071290584 = messageFormatterFn((function(  ) {
  return function(d) { return "Replace Recommendation..."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1497481254 = messageFormatterFn((function(  ) {
  return function(d) { return "Replace Recommendation"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a35212803 = messageFormatterFn((function(  ) {
  return function(d) { return "New Name"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1804484734 = messageFormatterFn((function(  ) {
  return function(d) { return "New Visualization"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1965773221 = messageFormatterFn((function(  ) {
  return function(d) { return "Recommendation"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a878938521 = messageFormatterFn((function(  ) {
  return function(d) { return "Select Recommendation"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b980850806 = messageFormatterFn((function(  ) {
  return function(d) { return "Replaced Recommended Visualization"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1872441010 = messageFormatterFn((function(  ) {
  return function(d) { return "Pin Visualization"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2138306859 = messageFormatterFn((function(  ) {
  return function(d) { return "Request Access"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a88765001 = messageFormatterFn((function(  ) {
  return function(d) { return "Error requesting access."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b833212727 = messageFormatterFn((function(  ) {
  return function(d) { return "An error occurred and the request couldn’t be sent."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1559110932 = messageFormatterFn((function(  ) {
  return function(d) { return "The owner will be notified"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a360075225 = messageFormatterFn((function(  ) {
  return function(d) { return "To see this page, request access from the owner."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a962839932 = messageFormatterFn((function(  ) {
  return function(d) { return "Request submitted"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a24389516 = messageFormatterFn((function(  ) {
  return function(d) { return "Restore " + d.numberOfSheets + " sheets from your previous session?"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1750677774 = messageFormatterFn((function(  ) {
  return function(d) { return "No"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a280863773 = messageFormatterFn((function(  ) {
  return function(d) { return "Restore Session"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1573500780 = messageFormatterFn((function(  ) {
  return function(d) { return "Yes"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b205768887 = messageFormatterFn((function(  ) {
  return function(d) { return "We’ve restored your previous session"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b204710269 = messageFormatterFn((function(  ) {
  return function(d) { return "Save as..."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1238331611 = messageFormatterFn((function(  ) {
  return function(d) { return "Save (overwrites)"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1169286255 = messageFormatterFn((function(  ) {
  return function(d) { return "Saved viz to workbook"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b479151311 = messageFormatterFn((function(  ) {
  return function(d) { return "Save..."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b826485424 = messageFormatterFn((function(  ) {
  return function(d) { return "Ask Data is unavailable at this time"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1491860036 = messageFormatterFn((function(  ) {
  return function(d) { return "Try and refresh your network connection."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1840830558 = messageFormatterFn((function(  ) {
  return function(d) { return "Reload this page."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1522644355 = messageFormatterFn((function(  ) {
  return function(d) { return "Either it doesn't exist, or you don't have permission to access it. Go to the " + d.datasourceName + " lens to ask more questions."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a155938350 = messageFormatterFn((function(  ) {
  return function(d) { return "We can't display that page."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1338711487 = messageFormatterFn((function(  ) {
  return function(d) { return "Share Ask Data Visualization"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1302609758 = messageFormatterFn((function(  ) {
  return function(d) { return "Sort"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2092952730 = messageFormatterFn((function(  ) {
  return function(d) { return "by"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1751564607 = messageFormatterFn((function(  ) {
  return function(d) { return "Alphabetical"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1906921349 = messageFormatterFn((function(  ) {
  return function(d) { return "Ascending"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a862664065 = messageFormatterFn((function(  ) {
  return function(d) { return "Descending"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b480018848 = messageFormatterFn((function(  ) {
  return function(d) { return "Error"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1507384193 = messageFormatterFn((function(  ) {
  return function(d) { return "Clear All"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1543066436 = messageFormatterFn((function(  ) {
  return function(d) { return "Embed Code"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1456520906 = messageFormatterFn((function(  ) {
  return function(d) { return "Redo"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1431059211 = messageFormatterFn((function(  ) {
  return function(d) { return "Save"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1322734953 = messageFormatterFn((function(  ) {
  return function(d) { return "Share Link"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1231702531 = messageFormatterFn((function(  ) {
  return function(d) { return "Share"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1517421168 = messageFormatterFn((function(  ) {
  return function(d) { return "Swap Axes"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b755331335 = messageFormatterFn((function(  ) {
  return function(d) { return "Please try again later."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a381839228 = messageFormatterFn((function(  ) {
  return function(d) { return "Undo"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2057660420 = messageFormatterFn((function(  ) {
  return function(d) { return "Ask Data is not supported for this datasource"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1182418277 = messageFormatterFn((function(  ) {
  return function(d) { return "View"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a168250261 = messageFormatterFn((function(  ) {
  return function(d) { return "No marks appear on this viz because a numeric field has null values."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1183221257 = messageFormatterFn((function(  ) {
  return function(d) { return "Bar Chart"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1232028600 = messageFormatterFn((function(  ) {
  return function(d) { return "Gantt Chart"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a750563440 = messageFormatterFn((function(  ) {
  return function(d) { return "Heat Map"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1062333512 = messageFormatterFn((function(  ) {
  return function(d) { return "Histogram"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a506389577 = messageFormatterFn((function(  ) {
  return function(d) { return "Line Chart"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1593632273 = messageFormatterFn((function(  ) {
  return function(d) { return "Map"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1170066128 = messageFormatterFn((function(  ) {
  return function(d) { return "Pie Chart"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a871133901 = messageFormatterFn((function(  ) {
  return function(d) { return "Scatter Plot"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1019078814 = messageFormatterFn((function(  ) {
  return function(d) { return "Stacked Bar"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b720589584 = messageFormatterFn((function(  ) {
  return function(d) { return "Text Table"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a572807074 = messageFormatterFn((function(  ) {
  return function(d) { return "Treemap"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a269362315 = messageFormatterFn((function(  ) {
  return function(d) { return "No marks appear on this map because geographic data is either missing or incorrectly structured."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b682748403 = messageFormatterFn((function(  ) {
  return function(d) { return "As Percentage"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b885045963 = messageFormatterFn((function(  ) {
  return function(d) { return "Calculate the difference of"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1118948133 = messageFormatterFn((function(  ) {
  return function(d) { return "Over the"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1942727245 = messageFormatterFn((function(  ) {
  return function(d) { return d.PROJECT_NAME; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a322493263 = messageFormatterFn((function(  ) {
  return function(d) { return "You do not have permission to move to “" + d.PROJECT_NAME + "”"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a849829004 = messageFormatterFn((function(  ) {
  return function(d) { return "Content cannot be moved to “" + d.PROJECT_NAME + "”"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b707568333 = messageFormatterFn((function(  ) {
  return function(d) { return d.SITE_NAME + " (site root)"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1994673257 = messageFormatterFn((function(  ) {
  return function(d) { return "Folders"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b109939126 = messageFormatterFn((function(  ) {
  return function(d) { return "Project hidden by permissions"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b558339907 = messageFormatterFn((function(  ) {
  return function(d) { return "Projects"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1421102511 = messageFormatterFn((function(  ) {
  return function(d) { return "Personal Space"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a623143930 = messageFormatterFn((function(  ) {
  return function(d) { return "(Private to me)"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a210862200 = messageFormatterFn((function(  ) {
  return function(d) { return "Search Projects"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b256468981 = messageFormatterFn((function(  ) {
  return function(d) { return "Cancel"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1133392248 = messageFormatterFn((function(  ) {
  return function(d) { return "Continue"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b290167621 = messageFormatterFn((function(  ) {
  return function(d) { return "Continue without signing in"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2021629502 = messageFormatterFn((function(  ) {
  return function(d) { return "Connect to " + d.DATA_SOURCE; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1323884826 = messageFormatterFn((function(  ) {
  return function(d) { return "Before you connect"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1942925832 = messageFormatterFn((function(  ) {
  return function(d) { return "Sign in with your " + d.DATA_SOURCE + " account to see the dashboard with your data, or continue without signing in to quickly see it with sample data."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1237574269 = messageFormatterFn((function(  ) {
  return function(d) { return "Make sure you’re using the cloud-based version of ServiceNow and have permission to access the necessary data. When you sign in, use your ServiceNow credentials, which may differ from those you use for single sign-on."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1673713816 = messageFormatterFn((function(  ) {
  return function(d) { return "You must be a company Administrator"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1663350618 = messageFormatterFn((function(  ) {
  return function(d) { return " on your QuickBooks Online account to use Tableau to connect to QuickBooks Online."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a69096519 = messageFormatterFn((function(  ) {
  return function(d) { return "Only one Company Administrator per company"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1663349657 = messageFormatterFn((function(  ) {
  return function(d) { return " can use Tableau to connect to QuickBooks Online. If another Company Administrator has used Tableau, that administrator must give up the Tableau Cloud application privilege so that you can use them."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b245754859 = messageFormatterFn((function(  ) {
  return function(d) { return "Sign in to " + d.DATA_SOURCE + " to see your data in the dashboard. Or continue without signing in to see sample data."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b852986918 = messageFormatterFn((function(  ) {
  return function(d) { return "Collections"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b324916712 = messageFormatterFn((function(  ) {
  return function(d) { return "Content types"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a150471517 = messageFormatterFn((function(  ) {
  return function(d) { return "Databases and Files"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b399310216 = messageFormatterFn((function(  ) {
  return function(d) { return "Data Roles"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1244463917 = messageFormatterFn((function(  ) {
  return function(d) { return "Data Sources"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b223222070 = messageFormatterFn((function(  ) {
  return function(d) { return "Flows"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1231001722 = messageFormatterFn((function(  ) {
  return function(d) { return "Hidden User"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1586755165 = messageFormatterFn((function(  ) {
  return function(d) { return "Ask Data Lenses"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1142034568 = messageFormatterFn((function(  ) {
  return function(d) { return "Metrics"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a273241859 = messageFormatterFn((function(  ) {
  return function(d) { return "Clear"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a827582142 = messageFormatterFn((function(  ) {
  return function(d) { return "Clear History"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1732264608 = messageFormatterFn((function(  ) {
  return function(d) { return "Your search history is private, and is only used to improve search suggestions."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b203645199 = messageFormatterFn((function(  ) {
  return function(d) { return "Clear all search history?"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2024936531 = messageFormatterFn((function(  ) {
  return function(d) { return "Recents"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1676771939 = messageFormatterFn((function( plural, pluralFuncs ) {
  return function(d) { return "Connected to " + d.NUMBER_LOC + " " + plural(d.NUMBER, 0, pluralFuncs.en, { one: function() { return "workbook";}, other: function() { return "workbooks";} }); }
})(messageFormat.plural, {"en": Globalize("en").pluralGenerator()}), Globalize("en").pluralGenerator({}));
Globalize.a1879712429 = messageFormatterFn((function(  ) {
  return function(d) { return "Searching."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1768549167 = messageFormatterFn((function(  ) {
  return function(d) { return "Search for views, metrics, workbooks, and more"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1196674041 = messageFormatterFn((function(  ) {
  return function(d) { return "Search for views, workbooks, and more"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a51946239 = messageFormatterFn((function( plural, pluralFuncs ) {
  return function(d) { return d.NUMBER_LOC + " " + plural(d.NUMBER, 0, pluralFuncs.en, { one: function() { return "item";}, other: function() { return "items";} }); }
})(messageFormat.plural, {"en": Globalize("en").pluralGenerator()}), Globalize("en").pluralGenerator({}));
Globalize.b1458676661 = messageFormatterFn((function(  ) {
  return function(d) { return "Press " + d.enter + " to see all"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2020721334 = messageFormatterFn((function(  ) {
  return function(d) { return "Enter"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2122073170 = messageFormatterFn((function( plural, pluralFuncs ) {
  return function(d) { return d.NUMBER_LOC + " recent " + plural(d.NUMBER, 0, pluralFuncs.en, { one: function() { return "view";}, other: function() { return "views";} }); }
})(messageFormat.plural, {"en": Globalize("en").pluralGenerator()}), Globalize("en").pluralGenerator({}));
Globalize.b932424467 = messageFormatterFn((function(  ) {
  return function(d) { return "Search results"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1463955377 = messageFormatterFn((function( plural, pluralFuncs ) {
  return function(d) { return "Modified " + d.NUMBER_LOC + " " + plural(d.NUMBER, 0, pluralFuncs.en, { one: function() { return "day";}, other: function() { return "days";} }) + " ago"; }
})(messageFormat.plural, {"en": Globalize("en").pluralGenerator()}), Globalize("en").pluralGenerator({}));
Globalize.b626169197 = messageFormatterFn((function( plural, pluralFuncs ) {
  return function(d) { return "Modified " + d.NUMBER_LOC + " " + plural(d.NUMBER, 0, pluralFuncs.en, { one: function() { return "hour";}, other: function() { return "hours";} }) + " ago"; }
})(messageFormat.plural, {"en": Globalize("en").pluralGenerator()}), Globalize("en").pluralGenerator({}));
Globalize.b2026412221 = messageFormatterFn((function( plural, pluralFuncs ) {
  return function(d) { return "Modified " + d.NUMBER_LOC + " " + plural(d.NUMBER, 0, pluralFuncs.en, { one: function() { return "minute";}, other: function() { return "minutes";} }) + " ago"; }
})(messageFormat.plural, {"en": Globalize("en").pluralGenerator()}), Globalize("en").pluralGenerator({}));
Globalize.a850293683 = messageFormatterFn((function( plural, pluralFuncs ) {
  return function(d) { return "Modified " + d.NUMBER_LOC + " " + plural(d.NUMBER, 0, pluralFuncs.en, { one: function() { return "month";}, other: function() { return "months";} }) + " ago"; }
})(messageFormat.plural, {"en": Globalize("en").pluralGenerator()}), Globalize("en").pluralGenerator({}));
Globalize.b992035421 = messageFormatterFn((function( plural, pluralFuncs ) {
  return function(d) { return "Modified " + d.NUMBER_LOC + " " + plural(d.NUMBER, 0, pluralFuncs.en, { one: function() { return "second";}, other: function() { return "seconds";} }) + " ago"; }
})(messageFormat.plural, {"en": Globalize("en").pluralGenerator()}), Globalize("en").pluralGenerator({}));
Globalize.a380097799 = messageFormatterFn((function(  ) {
  return function(d) { return "See all results for “" + d.SEARCH_TERM + "”"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2009835545 = messageFormatterFn((function(  ) {
  return function(d) { return "No results."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a11034216 = messageFormatterFn((function( plural, pluralFuncs ) {
  return function(d) { return "Showing " + d.TOTAL_COUNT_LOC + " " + plural(d.TOTAL_COUNT, 0, pluralFuncs.en, { one: function() { return "result";}, other: function() { return "results";} }) + "."; }
})(messageFormat.plural, {"en": Globalize("en").pluralGenerator()}), Globalize("en").pluralGenerator({}));
Globalize.b512163592 = messageFormatterFn((function(  ) {
  return function(d) { return "Owner"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b316901461 = messageFormatterFn((function(  ) {
  return function(d) { return "Projects"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1589939848 = messageFormatterFn((function(  ) {
  return function(d) { return "Virtual Connections"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b255502383 = messageFormatterFn((function(  ) {
  return function(d) { return "Simulations"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1255693226 = messageFormatterFn((function(  ) {
  return function(d) { return "Tables"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1798653261 = messageFormatterFn((function(  ) {
  return function(d) { return "Tables and Objects"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b667648794 = messageFormatterFn((function(  ) {
  return function(d) { return "Data Model Objects"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a923205184 = messageFormatterFn((function(  ) {
  return function(d) { return "Data Lake Objects"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a55994282 = messageFormatterFn((function(  ) {
  return function(d) { return "Calculated Insights"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a996812275 = messageFormatterFn((function(  ) {
  return function(d) { return "Views"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2046554626 = messageFormatterFn((function(  ) {
  return function(d) { return "Workbooks"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a276711850 = messageFormatterFn((function(  ) {
  return function(d) { return "Draft"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a687560591 = messageFormatterFn((function(  ) {
  return function(d) { return "You made changes to this workbook that have not been published. Your draft is visible only to you and system administrators."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1417128579 = messageFormatterFn((function(  ) {
  return function(d) { return "Never Published"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a829799334 = messageFormatterFn((function(  ) {
  return function(d) { return "This workbook has never been published and is only visible to you and system administrators."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1491193415 = messageFormatterFn((function(  ) {
  return function(d) { return "Drafts"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a368072565 = messageFormatterFn((function(  ) {
  return function(d) { return "Content suggestions"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1853334190 = messageFormatterFn((function(  ) {
  return function(d) { return "Having trouble?"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1874219758 = messageFormatterFn((function(  ) {
  return function(d) { return "You need Tableau Desktop version 2021.2 or higher to open this workbook. Install the latest version of Tableau Desktop or download the workbook as an earlier version."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a220194814 = messageFormatterFn((function(  ) {
  return function(d) { return "You need Tableau Desktop version 2021.2 or higher to open this workbook. Contact your administrator to install the latest version of Tableau Desktop, or download the workbook as an earlier version."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a830879600 = messageFormatterFn((function(  ) {
  return function(d) { return "Opening this file in Tableau Desktop. If it didn’t open, you can try again or continue editing in the browser."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b727368386 = messageFormatterFn((function(  ) {
  return function(d) { return "We recommend publishing your data source to Tableau Cloud or Server to maintain a single source for your data."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1145515008 = messageFormatterFn((function(  ) {
  return function(d) { return "Skip"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a120249559 = messageFormatterFn((function(  ) {
  return function(d) { return "Publish Data"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b337432452 = messageFormatterFn((function(  ) {
  return function(d) { return "Try Again"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1410244260 = messageFormatterFn((function(  ) {
  return function(d) { return "Opening in Tableau Desktop"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1848799075 = messageFormatterFn((function(  ) {
  return function(d) { return "Embedded Data"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1935844745 = messageFormatterFn((function(  ) {
  return function(d) { return "Playback Mode"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a99447117 = messageFormatterFn((function(  ) {
  return function(d) { return "Forward"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1413639702 = messageFormatterFn((function(  ) {
  return function(d) { return "Reverse"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a481873258 = messageFormatterFn((function(  ) {
  return function(d) { return "Stop"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a434350140 = messageFormatterFn((function(  ) {
  return function(d) { return "Previous"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a679036216 = messageFormatterFn((function(  ) {
  return function(d) { return "Next"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a572292111 = messageFormatterFn((function(  ) {
  return function(d) { return "Create Parameter"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b796057599 = messageFormatterFn((function(  ) {
  return function(d) { return "Edit Parameter"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b380121008 = messageFormatterFn((function(  ) {
  return function(d) { return "Name"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b965396537 = messageFormatterFn((function(  ) {
  return function(d) { return "Properties"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2028960704 = messageFormatterFn((function(  ) {
  return function(d) { return "Aliases"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1707846296 = messageFormatterFn((function(  ) {
  return function(d) { return "Data type"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1946800442 = messageFormatterFn((function(  ) {
  return function(d) { return "Float"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b762084937 = messageFormatterFn((function(  ) {
  return function(d) { return "Integer"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b663290207 = messageFormatterFn((function(  ) {
  return function(d) { return "String"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a803887649 = messageFormatterFn((function(  ) {
  return function(d) { return "Boolean"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1561467876 = messageFormatterFn((function(  ) {
  return function(d) { return "Date"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b147498921 = messageFormatterFn((function(  ) {
  return function(d) { return "DateTime"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1317822190 = messageFormatterFn((function(  ) {
  return function(d) { return "Spatial"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1290945164 = messageFormatterFn((function(  ) {
  return function(d) { return "Current value"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a589969604 = messageFormatterFn((function(  ) {
  return function(d) { return "Value when workbook opens"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1032591143 = messageFormatterFn((function(  ) {
  return function(d) { return "Allowable values"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a814593976 = messageFormatterFn((function(  ) {
  return function(d) { return "Range of values"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1793284123 = messageFormatterFn((function(  ) {
  return function(d) { return "All"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1156900099 = messageFormatterFn((function(  ) {
  return function(d) { return "List"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1704690096 = messageFormatterFn((function(  ) {
  return function(d) { return "Range"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1406788334 = messageFormatterFn((function(  ) {
  return function(d) { return "Add From Parameter"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1825521373 = messageFormatterFn((function(  ) {
  return function(d) { return "Add From Field"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b579381939 = messageFormatterFn((function(  ) {
  return function(d) { return "Paste From Clipboard"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b373888509 = messageFormatterFn((function(  ) {
  return function(d) { return "Set From Parameter"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1907494638 = messageFormatterFn((function(  ) {
  return function(d) { return "Set From Field"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b390040480 = messageFormatterFn((function(  ) {
  return function(d) { return "Cancel"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a896017954 = messageFormatterFn((function(  ) {
  return function(d) { return "OK"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a207983348 = messageFormatterFn((function(  ) {
  return function(d) { return "Years"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1202962175 = messageFormatterFn((function(  ) {
  return function(d) { return "Quarters"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b922461717 = messageFormatterFn((function(  ) {
  return function(d) { return "Months"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1563538467 = messageFormatterFn((function(  ) {
  return function(d) { return "Weeks"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b874578289 = messageFormatterFn((function(  ) {
  return function(d) { return "Days"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1689915411 = messageFormatterFn((function(  ) {
  return function(d) { return "Hours"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b359099139 = messageFormatterFn((function(  ) {
  return function(d) { return "Minutes"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1949817501 = messageFormatterFn((function(  ) {
  return function(d) { return "Seconds"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2102111346 = messageFormatterFn((function(  ) {
  return function(d) { return "ISO Weeks"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1777308961 = messageFormatterFn((function(  ) {
  return function(d) { return "ISO Quarters"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b421334135 = messageFormatterFn((function(  ) {
  return function(d) { return "ISO Years"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1044892251 = messageFormatterFn((function(  ) {
  return function(d) { return "Minimum"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a65546679 = messageFormatterFn((function(  ) {
  return function(d) { return "Maximum"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b300612582 = messageFormatterFn((function(  ) {
  return function(d) { return "Fixed"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1211901814 = messageFormatterFn((function(  ) {
  return function(d) { return "Step size"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1503408096 = messageFormatterFn((function(  ) {
  return function(d) { return "Value"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a503325244 = messageFormatterFn((function(  ) {
  return function(d) { return "Display As"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a433636253 = messageFormatterFn((function(  ) {
  return function(d) { return "Duplicate values found"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a332623435 = messageFormatterFn((function(  ) {
  return function(d) { return "Every value in the list must have a unique display name. Double-click the highlighted fields to update. Duplicate values will automatically be removed when you select OK."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b299983967 = messageFormatterFn((function(  ) {
  return function(d) { return d.alias + " is a duplicate"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1026962880 = messageFormatterFn((function(  ) {
  return function(d) { return "New row, press enter to edit"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1904455414 = messageFormatterFn((function(  ) {
  return function(d) { return "Remove Selected"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1188600784 = messageFormatterFn((function(  ) {
  return function(d) { return "Click to add"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b399982281 = messageFormatterFn((function(  ) {
  return function(d) { return "Your current browser doesn’t support this feature."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a192150811 = messageFormatterFn((function(  ) {
  return function(d) { return "When workbook opens"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1596972241 = messageFormatterFn((function(  ) {
  return function(d) { return "Add values from"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a114508620 = messageFormatterFn((function(  ) {
  return function(d) { return "None"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1622861122 = messageFormatterFn((function(  ) {
  return function(d) { return "Parameters"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b199740389 = messageFormatterFn((function(  ) {
  return function(d) { return "Display format"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a289262626 = messageFormatterFn((function(  ) {
  return function(d) { return "True"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a342529637 = messageFormatterFn((function(  ) {
  return function(d) { return "False"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b792858778 = messageFormatterFn((function(  ) {
  return function(d) { return "Domain Settings"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1627202100 = messageFormatterFn((function(  ) {
  return function(d) { return "No matches"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a39663799 = messageFormatterFn((function(  ) {
  return function(d) { return "Too many results to display"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b566787122 = messageFormatterFn((function(  ) {
  return function(d) { return "Publish Data Source"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1140649541 = messageFormatterFn((function(  ) {
  return function(d) { return "Project"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2077311227 = messageFormatterFn((function(  ) {
  return function(d) { return "Publish"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2126328496 = messageFormatterFn((function(  ) {
  return function(d) { return "Cancel"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1794622660 = messageFormatterFn((function(  ) {
  return function(d) { return "Close"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1759604447 = messageFormatterFn((function(  ) {
  return function(d) { return "Name"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1884351925 = messageFormatterFn((function(  ) {
  return function(d) { return "Update workbook to use the published data source"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1155836589 = messageFormatterFn((function(  ) {
  return function(d) { return "Name cannot be empty"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2067766151 = messageFormatterFn((function(  ) {
  return function(d) { return "Embed credentials"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a294117145 = messageFormatterFn((function(  ) {
  return function(d) { return "An unknown error occurred"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1991168594 = messageFormatterFn((function(  ) {
  return function(d) { return "Data Source Published Success"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1345151871 = messageFormatterFn((function(  ) {
  return function(d) { return "Go to the Data Source page to see this data source, change permissions, and other properties."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1008964057 = messageFormatterFn((function(  ) {
  return function(d) { return "Go to"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b716610594 = messageFormatterFn((function(  ) {
  return function(d) { return "This data source name already exists"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a810229873 = messageFormatterFn((function(  ) {
  return function(d) { return "Confirm Overwrite"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1640037685 = messageFormatterFn((function(  ) {
  return function(d) { return "There is already a data source named: “" + d.DS_NAME + "”."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2037243261 = messageFormatterFn((function(  ) {
  return function(d) { return "Are you sure you want to overwrite this data source?"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1039675090 = messageFormatterFn((function(  ) {
  return function(d) { return "OK"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1612360910 = messageFormatterFn((function(  ) {
  return function(d) { return "A project must be selected"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1976269297 = messageFormatterFn((function(  ) {
  return function(d) { return "You do not have permission to publish to any projects. Contact your Tableau Server administrator for access."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1476391612 = messageFormatterFn((function(  ) {
  return function(d) { return "Embed credentials to avoid errors when refreshing extracts and let users use this data source without being prompted for authentication."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b430435463 = messageFormatterFn((function(  ) {
  return function(d) { return "This data source contains calculations that reference different data sources in this workbook. These calculations won't be valid outside of this workbook."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a21021178 = messageFormatterFn((function(  ) {
  return function(d) { return "The extracted data source contains tables with user-filtered data. If you don't want this data source to include a user-specific view of the data, change its connection type from extract to live."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2044699256 = messageFormatterFn((function(  ) {
  return function(d) { return "Filter [" + d.fieldName + "]"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1234750006 = messageFormatterFn((function(  ) {
  return function(d) { return "Apply"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a443629738 = messageFormatterFn((function(  ) {
  return function(d) { return "OK"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1680551864 = messageFormatterFn((function(  ) {
  return function(d) { return "Cancel"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1169920041 = messageFormatterFn((function(  ) {
  return function(d) { return "Reset"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a470981337 = messageFormatterFn((function(  ) {
  return function(d) { return "Include null values"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1280907743 = messageFormatterFn((function(  ) {
  return function(d) { return "Relative dates"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a620028471 = messageFormatterFn((function(  ) {
  return function(d) { return "Range of dates"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a629155841 = messageFormatterFn((function(  ) {
  return function(d) { return "Starting date"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1353233606 = messageFormatterFn((function(  ) {
  return function(d) { return "Ending date"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1200566334 = messageFormatterFn((function(  ) {
  return function(d) { return "Range of values"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b95937618 = messageFormatterFn((function(  ) {
  return function(d) { return "At least"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b103243020 = messageFormatterFn((function(  ) {
  return function(d) { return "At most"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1566187905 = messageFormatterFn((function(  ) {
  return function(d) { return "Special"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1732872706 = messageFormatterFn((function(  ) {
  return function(d) { return "Set [" + d.fieldName + "] to show"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1507310459 = messageFormatterFn((function(  ) {
  return function(d) { return "Not a valid date"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b693635760 = messageFormatterFn((function(  ) {
  return function(d) { return "Not a valid value"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a877411657 = messageFormatterFn((function(  ) {
  return function(d) { return "Today"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2037899126 = messageFormatterFn((function(  ) {
  return function(d) { return "Anchor relative to"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1816190929 = messageFormatterFn((function(  ) {
  return function(d) { return "Show times"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b215093354 = messageFormatterFn((function(  ) {
  return function(d) { return "Minimum"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b987571928 = messageFormatterFn((function(  ) {
  return function(d) { return "Maximum"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a180614506 = messageFormatterFn((function(  ) {
  return function(d) { return "Only null values"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1936770101 = messageFormatterFn((function(  ) {
  return function(d) { return "Exclude null values"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a757620600 = messageFormatterFn((function(  ) {
  return function(d) { return "All values"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a801909715 = messageFormatterFn((function(  ) {
  return function(d) { return "Only null dates"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1010701650 = messageFormatterFn((function(  ) {
  return function(d) { return "Exclude null dates"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1119139835 = messageFormatterFn((function(  ) {
  return function(d) { return "All dates"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1309527155 = messageFormatterFn((function(  ) {
  return function(d) { return "Data Story"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b122245717 = messageFormatterFn((function(  ) {
  return function(d) { return "Add more fields"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2110947454 = messageFormatterFn((function(  ) {
  return function(d) { return "Remove Relationship"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1537890746 = messageFormatterFn((function(  ) {
  return function(d) { return "Select a field"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1695793938 = messageFormatterFn((function(  ) {
  return function(d) { return "Create Relationship Calculation..."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1390268932 = messageFormatterFn((function(  ) {
  return function(d) { return "Edit Calculation..."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2131395689 = messageFormatterFn((function(  ) {
  return function(d) { return "Learn more"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1478934583 = messageFormatterFn((function(  ) {
  return function(d) { return "Select matching fields to create this relationship."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1575542006 = messageFormatterFn((function(  ) {
  return function(d) { return "Operator"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1613707660 = messageFormatterFn((function(  ) {
  return function(d) { return "Performance Options"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a823216227 = messageFormatterFn((function(  ) {
  return function(d) { return "How do relationships differ from joins?"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1749093056 = messageFormatterFn((function(  ) {
  return function(d) { return "These settings help Tableau optimize queries during analysis. The default settings are recommended, if you aren't sure what to choose."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2104205069 = messageFormatterFn((function(  ) {
  return function(d) { return "Revert to Default"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a464381240 = messageFormatterFn((function(  ) {
  return function(d) { return "Change the setting if you know the cardinality."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1112816831 = messageFormatterFn((function(  ) {
  return function(d) { return "Tableau detected the cardinality for " + d.tableName + " from the database."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1211204037 = messageFormatterFn((function(  ) {
  return function(d) { return "Change the setting if you know the referential integrity."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1516352866 = messageFormatterFn((function(  ) {
  return function(d) { return "Tableau detected the referential integrity for " + d.tableName + " from the database."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2088717246 = messageFormatterFn((function(  ) {
  return function(d) { return "Cardinality"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1572549930 = messageFormatterFn((function(  ) {
  return function(d) { return "Referential Integrity"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1984214742 = messageFormatterFn((function(  ) {
  return function(d) { return "Many"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1180696027 = messageFormatterFn((function(  ) {
  return function(d) { return "One"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a216177474 = messageFormatterFn((function(  ) {
  return function(d) { return "Some records match"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b979573591 = messageFormatterFn((function(  ) {
  return function(d) { return "All records match"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2022622083 = messageFormatterFn((function(  ) {
  return function(d) { return "Values in the selected fields are unique."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1216114955 = messageFormatterFn((function(  ) {
  return function(d) { return "Values in " + d.fieldName + " are unique."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b632003457 = messageFormatterFn((function(  ) {
  return function(d) { return "This will join data before aggregation during analysis, optimizing queries. You might see duplicate aggregates if field values aren't unique."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a527331792 = messageFormatterFn((function(  ) {
  return function(d) { return "Tableau detected that values in the selected fields are unique."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1411206808 = messageFormatterFn((function(  ) {
  return function(d) { return "Tableau detected that values in " + d.fieldName + " are unique."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b742650380 = messageFormatterFn((function(  ) {
  return function(d) { return "Values in the selected fields aren't unique, or you don't know."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1923851196 = messageFormatterFn((function(  ) {
  return function(d) { return "Values in " + d.fieldName + " aren't unique, or you don't know."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1602132242 = messageFormatterFn((function(  ) {
  return function(d) { return "This will aggregate data before joins during analysis, preventing duplication of aggregates."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b826379683 = messageFormatterFn((function(  ) {
  return function(d) { return "Values in the selected fields in " + d.firstTable + " have a match in " + d.secondTable + "."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a547229285 = messageFormatterFn((function(  ) {
  return function(d) { return "Values in " + d.fieldName + " in " + d.firstTable + " have a match in " + d.secondTable + "."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1240316005 = messageFormatterFn((function(  ) {
  return function(d) { return "This will generate fewer and simpler joins to optimize queries. You might see inconsistent results during analysis if there are unmatched values in " + d.firstTable + "."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1066194998 = messageFormatterFn((function(  ) {
  return function(d) { return "Tableau detected that values in the selected fields in " + d.firstTable + " have a match in " + d.secondTable + "."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b45214082 = messageFormatterFn((function(  ) {
  return function(d) { return "Tableau detected that values in " + d.fieldName + " in " + d.firstTable + " have a match in " + d.secondTable + "."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1721365706 = messageFormatterFn((function(  ) {
  return function(d) { return "Some values in the selected fields in " + d.firstTable + " don't have a match in " + d.secondTable + ", or you don't know."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1671346158 = messageFormatterFn((function(  ) {
  return function(d) { return "Some values in " + d.fieldName + " in " + d.firstTable + " don't have a match in " + d.secondTable + ", or you don't know."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a958027608 = messageFormatterFn((function(  ) {
  return function(d) { return "This will keep all measure values, even unmatched ones."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b780000842 = messageFormatterFn((function(  ) {
  return function(d) { return "(detected)"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a792909133 = messageFormatterFn((function(  ) {
  return function(d) { return "(default)"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b621296824 = messageFormatterFn((function(  ) {
  return function(d) { return "Number (decimal)"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b605624534 = messageFormatterFn((function(  ) {
  return function(d) { return "Number (whole)"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a869801605 = messageFormatterFn((function(  ) {
  return function(d) { return "Date & Time"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1025239720 = messageFormatterFn((function(  ) {
  return function(d) { return "Date"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1704097477 = messageFormatterFn((function(  ) {
  return function(d) { return "String"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a960348052 = messageFormatterFn((function(  ) {
  return function(d) { return "Boolean"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a166656884 = messageFormatterFn((function(  ) {
  return function(d) { return "Set"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1561187984 = messageFormatterFn((function(  ) {
  return function(d) { return "Spatial"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a47742774 = messageFormatterFn((function(  ) {
  return function(d) { return "unknown"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b134386706 = messageFormatterFn((function(  ) {
  return function(d) { return "This relationship has no matching fields. Edit the relationship to select matching fields."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b304741588 = messageFormatterFn((function(  ) {
  return function(d) { return "Type mismatch between %1 (%2) and %3 (%4)."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1354473095 = messageFormatterFn((function(  ) {
  return function(d) { return "This relationship references unknown field %1. Edit the relationship to select a valid field."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a463443046 = messageFormatterFn((function(  ) {
  return function(d) { return "There is an error in one of the relationship links."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b59928283 = messageFormatterFn((function(  ) {
  return function(d) { return "There is an error in one of the inputs to the relationship."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1290600858 = messageFormatterFn((function(  ) {
  return function(d) { return "The relationship field %1 must only appear in a single relationship comparison."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b533741575 = messageFormatterFn((function(  ) {
  return function(d) { return "This relationship type is not supported."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2009780458 = messageFormatterFn((function(  ) {
  return function(d) { return "Relationship clause contains an invalid calculation."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b988357419 = messageFormatterFn((function(  ) {
  return function(d) { return "The related field %1 must be a spatial column to use a spatial comparison."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b898823708 = messageFormatterFn((function(  ) {
  return function(d) { return "The related field %1 is a spatial column and must use a spatial comparison."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1718750372 = messageFormatterFn((function(  ) {
  return function(d) { return "The relationship uses functionality not supported by extracts stored as 'Physical Tables.'"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b554944740 = messageFormatterFn((function(  ) {
  return function(d) { return "Unknown relationship error."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a249712115 = messageFormatterFn((function(  ) {
  return function(d) { return "Warning"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1854595545 = messageFormatterFn((function(  ) {
  return function(d) { return "Font family"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a379779484 = messageFormatterFn((function(  ) {
  return function(d) { return "Font size"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1907490471 = messageFormatterFn((function(  ) {
  return function(d) { return "Bold"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b31249444 = messageFormatterFn((function(  ) {
  return function(d) { return "Italic"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b469684444 = messageFormatterFn((function(  ) {
  return function(d) { return "Underline"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1807974662 = messageFormatterFn((function(  ) {
  return function(d) { return "Text Color"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1959191282 = messageFormatterFn((function(  ) {
  return function(d) { return "Left Align"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a329482816 = messageFormatterFn((function(  ) {
  return function(d) { return "Center Align"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1127363811 = messageFormatterFn((function(  ) {
  return function(d) { return "Right Align"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2050793046 = messageFormatterFn((function(  ) {
  return function(d) { return "is On"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a849750298 = messageFormatterFn((function(  ) {
  return function(d) { return "is Off"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2102935762 = messageFormatterFn((function(  ) {
  return function(d) { return "Link"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1177805006 = messageFormatterFn((function(  ) {
  return function(d) { return "Text"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a239848804 = messageFormatterFn((function(  ) {
  return function(d) { return "Insert"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1221186189 = messageFormatterFn((function(  ) {
  return function(d) { return "Insert field"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b912741772 = messageFormatterFn((function(  ) {
  return function(d) { return "Sheets"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2108764239 = messageFormatterFn((function(  ) {
  return function(d) { return "Clear Formatting"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b489782882 = messageFormatterFn((function(  ) {
  return function(d) { return "Clear Formatting"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1214654835 = messageFormatterFn((function(  ) {
  return function(d) { return "Apply"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b735460926 = messageFormatterFn((function(  ) {
  return function(d) { return "Apply"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2074531459 = messageFormatterFn((function(  ) {
  return function(d) { return "Cancel"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a972253906 = messageFormatterFn((function(  ) {
  return function(d) { return "Cancel"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1533963233 = messageFormatterFn((function(  ) {
  return function(d) { return "OK"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1177231536 = messageFormatterFn((function(  ) {
  return function(d) { return "OK"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a754311282 = messageFormatterFn((function(  ) {
  return function(d) { return "Reset"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b743983871 = messageFormatterFn((function(  ) {
  return function(d) { return "Reset"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2118455319 = messageFormatterFn((function(  ) {
  return function(d) { return "Hyperlink"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a191283896 = messageFormatterFn((function(  ) {
  return function(d) { return "Hyperlink"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1691969263 = messageFormatterFn((function(  ) {
  return function(d) { return "Edit"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a224624104 = messageFormatterFn((function(  ) {
  return function(d) { return "Remove Link"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2115329623 = messageFormatterFn((function(  ) {
  return function(d) { return "Remove"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1524583846 = messageFormatterFn((function(  ) {
  return function(d) { return "Remove"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1245571961 = messageFormatterFn((function(  ) {
  return function(d) { return "Rich text editor"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a911272127 = messageFormatterFn((function(  ) {
  return function(d) { return "Show tooltips"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b602924848 = messageFormatterFn((function(  ) {
  return function(d) { return "Include command buttons"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1455012986 = messageFormatterFn((function(  ) {
  return function(d) { return "Allow selection by category"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a359365104 = messageFormatterFn((function(  ) {
  return function(d) { return "Responsive - Show tooltips instantly"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a168138069 = messageFormatterFn((function(  ) {
  return function(d) { return "On Hover - Show tooltips on hover"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a121743442 = messageFormatterFn((function(  ) {
  return function(d) { return "at " + d.TIME; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a661070208 = messageFormatterFn((function(  ) {
  return function(d) { return d.EVERY_DAYS_OF_THE_WEEK + ", " + d.RECURRING_EVERY + " " + d.STARTING_FROM_ENDING_AT; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a950063681 = messageFormatterFn((function(  ) {
  return function(d) { return d.EVERY_DAYS_OF_WEEK + ", " + d.AT; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a181374051 = messageFormatterFn((function(  ) {
  return function(d) { return "Every " + d.DAYS_OF_MONTH + ", " + d.AT; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1204242852 = messageFormatterFn((function(  ) {
  return function(d) { return d.ORDINAL + " " + d.DAY_OF_WEEK; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1211162 = messageFormatterFn((function(  ) {
  return function(d) { return d.DAYS_OF_MONTH + " day of the month"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a606373643 = messageFormatterFn((function(  ) {
  return function(d) { return "day of the month"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1487624747 = messageFormatterFn((function(  ) {
  return function(d) { return "last"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2091160741 = messageFormatterFn((function(  ) {
  return function(d) { return "Fifth"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2080096748 = messageFormatterFn((function(  ) {
  return function(d) { return "First"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1056594508 = messageFormatterFn((function(  ) {
  return function(d) { return "Fourth"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b932890543 = messageFormatterFn((function(  ) {
  return function(d) { return "Friday"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b872516216 = messageFormatterFn((function(  ) {
  return function(d) { return "Last"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a157481922 = messageFormatterFn((function(  ) {
  return function(d) { return "Monday"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b384423317 = messageFormatterFn((function(  ) {
  return function(d) { return "Saturday"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1434912294 = messageFormatterFn((function(  ) {
  return function(d) { return "Second"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1240503166 = messageFormatterFn((function(  ) {
  return function(d) { return "Sunday"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1718034187 = messageFormatterFn((function(  ) {
  return function(d) { return "Third"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1239287732 = messageFormatterFn((function(  ) {
  return function(d) { return "Thursday"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b574370767 = messageFormatterFn((function(  ) {
  return function(d) { return "Tuesday"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1392454470 = messageFormatterFn((function(  ) {
  return function(d) { return "Wednesday"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b780766882 = messageFormatterFn((function(  ) {
  return function(d) { return "1ˢᵗ"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b780007692 = messageFormatterFn((function(  ) {
  return function(d) { return "2ⁿᵈ"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b778965007 = messageFormatterFn((function(  ) {
  return function(d) { return "3ʳᵈ"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b753966514 = messageFormatterFn((function(  ) {
  return function(d) { return d.N + "ᵗʰ"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1232555827 = messageFormatterFn((function(  ) {
  return function(d) { return "At"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1008366748 = messageFormatterFn((function(  ) {
  return function(d) { return "Done"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2020303977 = messageFormatterFn((function(  ) {
  return function(d) { return "Every"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1241100250 = messageFormatterFn((function(  ) {
  return function(d) { return "All days of the week"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1068433092 = messageFormatterFn((function(  ) {
  return function(d) { return "From"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1232144519 = messageFormatterFn((function(  ) {
  return function(d) { return "On"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b991252684 = messageFormatterFn((function(  ) {
  return function(d) { return "Repeats"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1231994603 = messageFormatterFn((function(  ) {
  return function(d) { return "To"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1239665325 = messageFormatterFn((function(  ) {
  return function(d) { return "Daily"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a957272954 = messageFormatterFn((function(  ) {
  return function(d) { return "Friday"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2042905231 = messageFormatterFn((function(  ) {
  return function(d) { return "Fri"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1469360696 = messageFormatterFn((function(  ) {
  return function(d) { return "F"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2047645419 = messageFormatterFn((function(  ) {
  return function(d) { return "Monday"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a29617472 = messageFormatterFn((function(  ) {
  return function(d) { return "Mon"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1366530713 = messageFormatterFn((function(  ) {
  return function(d) { return "M"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b708468908 = messageFormatterFn((function(  ) {
  return function(d) { return "Saturday"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b754629911 = messageFormatterFn((function(  ) {
  return function(d) { return "Sat"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a409286302 = messageFormatterFn((function(  ) {
  return function(d) { return "Sa"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a649660331 = messageFormatterFn((function(  ) {
  return function(d) { return "Sunday"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b469185024 = messageFormatterFn((function(  ) {
  return function(d) { return "Sun"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1942514855 = messageFormatterFn((function(  ) {
  return function(d) { return "Su"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1563333323 = messageFormatterFn((function(  ) {
  return function(d) { return "Thursday"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1201497098 = messageFormatterFn((function(  ) {
  return function(d) { return "Thu"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a195292509 = messageFormatterFn((function(  ) {
  return function(d) { return "Th"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2108844504 = messageFormatterFn((function(  ) {
  return function(d) { return "Tuesday"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b907687235 = messageFormatterFn((function(  ) {
  return function(d) { return "Tue"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1512727626 = messageFormatterFn((function(  ) {
  return function(d) { return "T"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1447034097 = messageFormatterFn((function(  ) {
  return function(d) { return "Wednesday"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1531431226 = messageFormatterFn((function(  ) {
  return function(d) { return "Wed"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1832529119 = messageFormatterFn((function(  ) {
  return function(d) { return "W"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2049845105 = messageFormatterFn((function(  ) {
  return function(d) { return "Every " + d.DAYS_OF_WEEK_LOC; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b945707469 = messageFormatterFn((function(  ) {
  return function(d) { return "every hour"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a747915638 = messageFormatterFn((function(  ) {
  return function(d) { return "every " + d.NUM_HOURS + " hours"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1639362630 = messageFormatterFn((function(  ) {
  return function(d) { return "every " + d.NUM_MINUTES + " minutes"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1954838669 = messageFormatterFn((function(  ) {
  return function(d) { return "Hourly"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1833563254 = messageFormatterFn((function(  ) {
  return function(d) { return "Eight hours"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1451041813 = messageFormatterFn((function(  ) {
  return function(d) { return "Fifteen minutes"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b128945283 = messageFormatterFn((function(  ) {
  return function(d) { return "Four hours"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1552946998 = messageFormatterFn((function(  ) {
  return function(d) { return "Half hour"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b188687689 = messageFormatterFn((function(  ) {
  return function(d) { return "Hour"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2119786359 = messageFormatterFn((function(  ) {
  return function(d) { return "Six hours"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b950669136 = messageFormatterFn((function(  ) {
  return function(d) { return "Twelve hours"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b400331149 = messageFormatterFn((function(  ) {
  return function(d) { return "Two hours"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1757411247 = messageFormatterFn((function(  ) {
  return function(d) { return "eight hours"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1932592238 = messageFormatterFn((function(  ) {
  return function(d) { return "fifteen minutes"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a807043370 = messageFormatterFn((function(  ) {
  return function(d) { return "four hours"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1566199171 = messageFormatterFn((function(  ) {
  return function(d) { return "half hour"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a382835504 = messageFormatterFn((function(  ) {
  return function(d) { return "hour"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a601375134 = messageFormatterFn((function(  ) {
  return function(d) { return "six hours"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b40686249 = messageFormatterFn((function(  ) {
  return function(d) { return "twelve hours"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1670609652 = messageFormatterFn((function(  ) {
  return function(d) { return "two hours"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1512873313 = messageFormatterFn((function(  ) {
  return function(d) { return "Monthly"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1506980252 = messageFormatterFn((function(  ) {
  return function(d) { return "Day of week"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1779887559 = messageFormatterFn((function(  ) {
  return function(d) { return "Days of month"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b496796665 = messageFormatterFn((function(  ) {
  return function(d) { return "of the month"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1222970931 = messageFormatterFn((function(  ) {
  return function(d) { return "Fifth"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1211906938 = messageFormatterFn((function(  ) {
  return function(d) { return "First"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2094292454 = messageFormatterFn((function(  ) {
  return function(d) { return "Fourth"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1814341418 = messageFormatterFn((function(  ) {
  return function(d) { return "Last"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1715974668 = messageFormatterFn((function(  ) {
  return function(d) { return "Second"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1708743299 = messageFormatterFn((function(  ) {
  return function(d) { return "Third"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1245593293 = messageFormatterFn((function(  ) {
  return function(d) { return "time a month"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a254545914 = messageFormatterFn((function(  ) {
  return function(d) { return "times a month"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b656595800 = messageFormatterFn((function(  ) {
  return function(d) { return d.WHICH + d.WHAT + " " + d.OF_THE_MONTH + " at " + d.WHEN; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a509739462 = messageFormatterFn((function(  ) {
  return function(d) { return "On Data Refresh"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1545015087 = messageFormatterFn((function(  ) {
  return function(d) { return "Day"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a332907094 = messageFormatterFn((function(  ) {
  return function(d) { return "day"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a347068190 = messageFormatterFn((function(  ) {
  return function(d) { return "Specified Time"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1695779221 = messageFormatterFn((function(  ) {
  return function(d) { return "starting at " + d.START_AT; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a850303978 = messageFormatterFn((function(  ) {
  return function(d) { return d.NUM_DAYS + " " + d.DAYS + " " + d.AT_OR_FROM_TO; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b912413145 = messageFormatterFn((function(  ) {
  return function(d) { return d.NUM_DAYS + " " + d.DAYS + " every " + d.LOC_MIN_NUM + " " + d.AT_OR_FROM_TO; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1384732830 = messageFormatterFn((function(  ) {
  return function(d) { return "day a week,"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1527914641 = messageFormatterFn((function(  ) {
  return function(d) { return "days a week,"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1492433292 = messageFormatterFn((function(  ) {
  return function(d) { return "at " + d.START_TIME; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a397210014 = messageFormatterFn((function(  ) {
  return function(d) { return "from " + d.START_TIME + " to " + d.END_TIME; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b455527767 = messageFormatterFn((function(  ) {
  return function(d) { return d["1"] + " at " + d["2"]; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a965394059 = messageFormatterFn((function(  ) {
  return function(d) { return d.NUM_DAYS + " " + d.DAYS + " every " + d.LOC_MIN_NUM + " from " + d.START_TIME + " to " + d.END_TIME; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1288108565 = messageFormatterFn((function(  ) {
  return function(d) { return d.NUM_DAYS + " " + d.DAYS + " every " + d.LOC_MIN_NUM + " starting at " + d.START_TIME; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2076107680 = messageFormatterFn((function(  ) {
  return function(d) { return "to"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2035136165 = messageFormatterFn((function(  ) {
  return function(d) { return "Time zone"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1962846595 = messageFormatterFn((function(  ) {
  return function(d) { return "Weekly"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b936157242 = messageFormatterFn((function(  ) {
  return function(d) { return d.NUM_DAYS_LOC + " " + d.DAYS + " at " + d.START_TIME; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a352825700 = messageFormatterFn((function(  ) {
  return function(d) { return "Clear"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b861571267 = messageFormatterFn((function(  ) {
  return function(d) { return "collection"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b384955895 = messageFormatterFn((function(  ) {
  return function(d) { return "content type: " + d.CONTENT_TYPE; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a912628334 = messageFormatterFn((function(  ) {
  return function(d) { return "Copy Link"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a670480718 = messageFormatterFn((function(  ) {
  return function(d) { return "Copy Link"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a623471103 = messageFormatterFn((function(  ) {
  return function(d) { return "Copy Site ID"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a511273764 = messageFormatterFn((function(  ) {
  return function(d) { return "data source"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1831354759 = messageFormatterFn((function(  ) {
  return function(d) { return "\"" + d.CONTENT_NAME + "\" embed code copied to the clipboard."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b294899597 = messageFormatterFn((function(  ) {
  return function(d) { return "Embed Code"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1538946345 = messageFormatterFn((function(  ) {
  return function(d) { return "Copy Embed Code"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2037248205 = messageFormatterFn((function(  ) {
  return function(d) { return "flow"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b707595035 = messageFormatterFn((function(  ) {
  return function(d) { return "Can't share because the list of users contains errors. Correct the errors, then try again."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a468534997 = messageFormatterFn((function(  ) {
  return function(d) { return "\"" + d.CONTENT_NAME + "\" link copied to the clipboard."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b363498191 = messageFormatterFn((function(  ) {
  return function(d) { return "Share using a link"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a147001133 = messageFormatterFn((function(  ) {
  return function(d) { return "Getting link…"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b575829006 = messageFormatterFn((function(  ) {
  return function(d) { return "Enter at least one username before sharing."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1126154479 = messageFormatterFn((function(  ) {
  return function(d) { return d.NAME + " doesn't have an email address set to receive notifications."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1746986670 = messageFormatterFn((function(  ) {
  return function(d) { return d.NAME + " isn't a user on this site."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b771271794 = messageFormatterFn((function(  ) {
  return function(d) { return "Add username to share"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a638096484 = messageFormatterFn((function(  ) {
  return function(d) { return d.FULL_NAME + " (" + d.USER_NAME + ")"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a140655664 = messageFormatterFn((function(  ) {
  return function(d) { return "project"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a107260355 = messageFormatterFn((function(  ) {
  return function(d) { return "virtual connection"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1406472184 = messageFormatterFn((function(  ) {
  return function(d) { return d.USER_FRIENDLY_NAME + " requested access to “" + d.CONTENT_NAME + "”. Do you want to grant this user view access?"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1550265974 = messageFormatterFn((function(  ) {
  return function(d) { return "Share"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a825958650 = messageFormatterFn((function(  ) {
  return function(d) { return "Dismiss"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1349592048 = messageFormatterFn((function(  ) {
  return function(d) { return "Grant Access"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1409747579 = messageFormatterFn((function( plural, pluralFuncs ) {
  return function(d) { return "Failed to grant access for " + plural(d.USER_COUNT, 0, pluralFuncs.en, { one: function() { return d.USER_COUNT_LOC + " user";}, other: function() { return d.USER_COUNT_LOC + " users";} }) + "."; }
})(messageFormat.plural, {"en": Globalize("en").pluralGenerator()}), Globalize("en").pluralGenerator({}));
Globalize.a182981674 = messageFormatterFn((function( plural, pluralFuncs ) {
  return function(d) { return "Access granted to " + plural(d.USER_COUNT, 0, pluralFuncs.en, { one: function() { return d.USER_COUNT_LOC + " user";}, other: function() { return d.USER_COUNT_LOC + " users";} }) + "."; }
})(messageFormat.plural, {"en": Globalize("en").pluralGenerator()}), Globalize("en").pluralGenerator({}));
Globalize.b82093403 = messageFormatterFn((function(  ) {
  return function(d) { return "Manage Permissions"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1769228886 = messageFormatterFn((function(  ) {
  return function(d) { return "Grant view access to all datasources in project “" + d.PROJECT_NAME + "”?"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b711714861 = messageFormatterFn((function(  ) {
  return function(d) { return "Grant view access to all flows in project “" + d.PROJECT_NAME + "”?"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a774648149 = messageFormatterFn((function(  ) {
  return function(d) { return "Grant view access to all metrics in project “" + d.PROJECT_NAME + "”?"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2076610710 = messageFormatterFn((function(  ) {
  return function(d) { return "Grant view access to all projects in project “" + d.PROJECT_NAME + "”?"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b256716918 = messageFormatterFn((function(  ) {
  return function(d) { return "Grant view access to all views in project “" + d.PROJECT_NAME + "”?"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1856250598 = messageFormatterFn((function(  ) {
  return function(d) { return "Grant view access to all virtual connections in project “" + d.PROJECT_NAME + "”?"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1547460639 = messageFormatterFn((function(  ) {
  return function(d) { return "Grant view access to all workbooks in project “" + d.PROJECT_NAME + "”?"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b873160959 = messageFormatterFn((function(  ) {
  return function(d) { return "Grant view access to all views in workbook “" + d.WORKBOOK_NAME + "”?"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a818822554 = messageFormatterFn((function(  ) {
  return function(d) { return "Grant view access to datasource “" + d.CONTENT_NAME + "”?"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b694725693 = messageFormatterFn((function(  ) {
  return function(d) { return "Grant view access to flow “" + d.CONTENT_NAME + "”?"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b78630587 = messageFormatterFn((function(  ) {
  return function(d) { return "Grant view access to metric “" + d.CONTENT_NAME + "”?"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1536519546 = messageFormatterFn((function(  ) {
  return function(d) { return "Grant view access to project “" + d.CONTENT_NAME + "”?"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b239727750 = messageFormatterFn((function(  ) {
  return function(d) { return "Grant view access to view “" + d.CONTENT_NAME + "”?"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1788708650 = messageFormatterFn((function(  ) {
  return function(d) { return "Grant view access to virtual connection “" + d.CONTENT_NAME + "”?"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1885348879 = messageFormatterFn((function(  ) {
  return function(d) { return "Grant view access to workbook “" + d.CONTENT_NAME + "”?"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a403848633 = messageFormatterFn((function(  ) {
  return function(d) { return "locked permissions"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a39739553 = messageFormatterFn((function(  ) {
  return function(d) { return "tabbed views"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1538678468 = messageFormatterFn((function( plural, pluralFuncs ) {
  return function(d) { return "“" + d.CONTENT_NAME + "” shared with " + plural(d.TOTAL_USER_COUNT, 0, pluralFuncs.en, { one: function() { return d.TOTAL_USER_COUNT_LOC + " user";}, other: function() { return d.TOTAL_USER_COUNT_LOC + " users";} }) + ". " + plural(d.UNPERMISSIONED_USER_COUNT, 0, pluralFuncs.en, { one: function() { return d.UNPERMISSIONED_USER_COUNT_LOC;}, other: function() { return d.UNPERMISSIONED_USER_COUNT_LOC;} }) + " of these users don't have permission to view the collection. Do you want to grant these users view access?"; }
})(messageFormat.plural, {"en": Globalize("en").pluralGenerator()}), Globalize("en").pluralGenerator({}));
Globalize.b165833437 = messageFormatterFn((function( plural, pluralFuncs ) {
  return function(d) { return "“" + d.CONTENT_NAME + "” shared with " + plural(d.TOTAL_USER_COUNT, 0, pluralFuncs.en, { one: function() { return d.TOTAL_USER_COUNT_LOC + " user";}, other: function() { return d.TOTAL_USER_COUNT_LOC + " users";} }) + ". " + plural(d.UNPERMISSIONED_USER_COUNT, 0, pluralFuncs.en, { one: function() { return d.UNPERMISSIONED_USER_COUNT_LOC;}, other: function() { return d.UNPERMISSIONED_USER_COUNT_LOC;} }) + " of these users don't have permission to view the datasource. Do you want to grant these users the view permissions template?"; }
})(messageFormat.plural, {"en": Globalize("en").pluralGenerator()}), Globalize("en").pluralGenerator({}));
Globalize.b1040366068 = messageFormatterFn((function( plural, pluralFuncs ) {
  return function(d) { return "“" + d.CONTENT_NAME + "” shared with " + plural(d.TOTAL_USER_COUNT, 0, pluralFuncs.en, { one: function() { return d.TOTAL_USER_COUNT_LOC + " user";}, other: function() { return d.TOTAL_USER_COUNT_LOC + " users";} }) + ". " + plural(d.UNPERMISSIONED_USER_COUNT, 0, pluralFuncs.en, { one: function() { return d.UNPERMISSIONED_USER_COUNT_LOC;}, other: function() { return d.UNPERMISSIONED_USER_COUNT_LOC;} }) + " of these users don't have permission to view the flow. Do you want to grant these users the view permissions template?"; }
})(messageFormat.plural, {"en": Globalize("en").pluralGenerator()}), Globalize("en").pluralGenerator({}));
Globalize.b1526549170 = messageFormatterFn((function( plural, pluralFuncs ) {
  return function(d) { return "“" + d.CONTENT_NAME + "” shared with " + plural(d.TOTAL_USER_COUNT, 0, pluralFuncs.en, { one: function() { return d.TOTAL_USER_COUNT_LOC + " user";}, other: function() { return d.TOTAL_USER_COUNT_LOC + " users";} }) + ". " + plural(d.UNPERMISSIONED_USER_COUNT, 0, pluralFuncs.en, { one: function() { return d.UNPERMISSIONED_USER_COUNT_LOC;}, other: function() { return d.UNPERMISSIONED_USER_COUNT_LOC;} }) + " of these users don't have permission to view the metric. Do you want to grant these users the view permissions template?"; }
})(messageFormat.plural, {"en": Globalize("en").pluralGenerator()}), Globalize("en").pluralGenerator({}));
Globalize.b399283567 = messageFormatterFn((function( plural, pluralFuncs ) {
  return function(d) { return "“" + d.CONTENT_NAME + "” shared with " + plural(d.TOTAL_USER_COUNT, 0, pluralFuncs.en, { one: function() { return d.TOTAL_USER_COUNT_LOC + " user";}, other: function() { return d.TOTAL_USER_COUNT_LOC + " users";} }) + ". " + plural(d.UNPERMISSIONED_USER_COUNT, 0, pluralFuncs.en, { one: function() { return d.UNPERMISSIONED_USER_COUNT_LOC;}, other: function() { return d.UNPERMISSIONED_USER_COUNT_LOC;} }) + " of these users don't have permission to view the project. Do you want to grant these users the view permissions template?"; }
})(messageFormat.plural, {"en": Globalize("en").pluralGenerator()}), Globalize("en").pluralGenerator({}));
Globalize.b585368125 = messageFormatterFn((function( plural, pluralFuncs ) {
  return function(d) { return "“" + d.CONTENT_NAME + "” shared with " + plural(d.TOTAL_USER_COUNT, 0, pluralFuncs.en, { one: function() { return d.TOTAL_USER_COUNT_LOC + " user";}, other: function() { return d.TOTAL_USER_COUNT_LOC + " users";} }) + ". " + plural(d.UNPERMISSIONED_USER_COUNT, 0, pluralFuncs.en, { one: function() { return d.UNPERMISSIONED_USER_COUNT_LOC;}, other: function() { return d.UNPERMISSIONED_USER_COUNT_LOC;} }) + " of these users don't have permission to view the view. Do you want to grant these users the view permissions template?"; }
})(messageFormat.plural, {"en": Globalize("en").pluralGenerator()}), Globalize("en").pluralGenerator({}));
Globalize.b64007295 = messageFormatterFn((function( plural, pluralFuncs ) {
  return function(d) { return "“" + d.CONTENT_NAME + "” shared with " + plural(d.TOTAL_USER_COUNT, 0, pluralFuncs.en, { one: function() { return d.TOTAL_USER_COUNT_LOC + " user";}, other: function() { return d.TOTAL_USER_COUNT_LOC + " users";} }) + ". " + plural(d.UNPERMISSIONED_USER_COUNT, 0, pluralFuncs.en, { one: function() { return d.UNPERMISSIONED_USER_COUNT_LOC;}, other: function() { return d.UNPERMISSIONED_USER_COUNT_LOC;} }) + " of these users don't have permission to view the virtual connection. Do you want to grant these users the view permissions template?"; }
})(messageFormat.plural, {"en": Globalize("en").pluralGenerator()}), Globalize("en").pluralGenerator({}));
Globalize.a2004994520 = messageFormatterFn((function( plural, pluralFuncs ) {
  return function(d) { return "“" + d.CONTENT_NAME + "” shared with " + plural(d.TOTAL_USER_COUNT, 0, pluralFuncs.en, { one: function() { return d.TOTAL_USER_COUNT_LOC + " user";}, other: function() { return d.TOTAL_USER_COUNT_LOC + " users";} }) + ". " + plural(d.UNPERMISSIONED_USER_COUNT, 0, pluralFuncs.en, { one: function() { return d.UNPERMISSIONED_USER_COUNT_LOC;}, other: function() { return d.UNPERMISSIONED_USER_COUNT_LOC;} }) + " of these users don't have permission to view the workbook. Do you want to grant these users the view permissions template?"; }
})(messageFormat.plural, {"en": Globalize("en").pluralGenerator()}), Globalize("en").pluralGenerator({}));
Globalize.a220375411 = messageFormatterFn((function( plural, pluralFuncs ) {
  return function(d) { return "“" + d.CONTENT_NAME + "” shared with " + plural(d.TOTAL_USER_COUNT, 0, pluralFuncs.en, { one: function() { return d.TOTAL_USER_COUNT_LOC + " user";}, other: function() { return d.TOTAL_USER_COUNT_LOC + " users";} }) + ". " + plural(d.UNPERMISSIONED_USER_COUNT, 0, pluralFuncs.en, { one: function() { return d.UNPERMISSIONED_USER_COUNT_LOC;}, other: function() { return d.UNPERMISSIONED_USER_COUNT_LOC;} }) + " of these users don't have permission to view the datasource. This datasource is in a project with [LOCKED_PERMISSIONS] and can't be managed individually."; }
})(messageFormat.plural, {"en": Globalize("en").pluralGenerator()}), Globalize("en").pluralGenerator({}));
Globalize.a1983632988 = messageFormatterFn((function( plural, pluralFuncs ) {
  return function(d) { return "“" + d.CONTENT_NAME + "” shared with " + plural(d.TOTAL_USER_COUNT, 0, pluralFuncs.en, { one: function() { return d.TOTAL_USER_COUNT_LOC + " user";}, other: function() { return d.TOTAL_USER_COUNT_LOC + " users";} }) + ". " + plural(d.UNPERMISSIONED_USER_COUNT, 0, pluralFuncs.en, { one: function() { return d.UNPERMISSIONED_USER_COUNT_LOC;}, other: function() { return d.UNPERMISSIONED_USER_COUNT_LOC;} }) + " of these users don't have permission to view the flow. This flow is in a project with [LOCKED_PERMISSIONS] and can't be managed individually."; }
})(messageFormat.plural, {"en": Globalize("en").pluralGenerator()}), Globalize("en").pluralGenerator({}));
Globalize.a1138651550 = messageFormatterFn((function( plural, pluralFuncs ) {
  return function(d) { return "“" + d.CONTENT_NAME + "” shared with " + plural(d.TOTAL_USER_COUNT, 0, pluralFuncs.en, { one: function() { return d.TOTAL_USER_COUNT_LOC + " user";}, other: function() { return d.TOTAL_USER_COUNT_LOC + " users";} }) + ". " + plural(d.UNPERMISSIONED_USER_COUNT, 0, pluralFuncs.en, { one: function() { return d.UNPERMISSIONED_USER_COUNT_LOC;}, other: function() { return d.UNPERMISSIONED_USER_COUNT_LOC;} }) + " of these users don't have permission to view the metric. This metric is in a project with [LOCKED_PERMISSIONS] and can't be managed individually."; }
})(messageFormat.plural, {"en": Globalize("en").pluralGenerator()}), Globalize("en").pluralGenerator({}));
Globalize.a617560129 = messageFormatterFn((function( plural, pluralFuncs ) {
  return function(d) { return "“" + d.CONTENT_NAME + "” shared with " + plural(d.TOTAL_USER_COUNT, 0, pluralFuncs.en, { one: function() { return d.TOTAL_USER_COUNT_LOC + " user";}, other: function() { return d.TOTAL_USER_COUNT_LOC + " users";} }) + ". " + plural(d.UNPERMISSIONED_USER_COUNT, 0, pluralFuncs.en, { one: function() { return d.UNPERMISSIONED_USER_COUNT_LOC;}, other: function() { return d.UNPERMISSIONED_USER_COUNT_LOC;} }) + " of these users don't have permission to view the project. This project is in a project with [LOCKED_PERMISSIONS] and can't be managed individually."; }
})(messageFormat.plural, {"en": Globalize("en").pluralGenerator()}), Globalize("en").pluralGenerator({}));
Globalize.b1856336365 = messageFormatterFn((function( plural, pluralFuncs ) {
  return function(d) { return "“" + d.CONTENT_NAME + "” shared with " + plural(d.TOTAL_USER_COUNT, 0, pluralFuncs.en, { one: function() { return d.TOTAL_USER_COUNT_LOC + " user";}, other: function() { return d.TOTAL_USER_COUNT_LOC + " users";} }) + ". " + plural(d.UNPERMISSIONED_USER_COUNT, 0, pluralFuncs.en, { one: function() { return d.UNPERMISSIONED_USER_COUNT_LOC;}, other: function() { return d.UNPERMISSIONED_USER_COUNT_LOC;} }) + " of these users don't have permission to view the view. This view is in a project with [LOCKED_PERMISSIONS] and can't be managed individually."; }
})(messageFormat.plural, {"en": Globalize("en").pluralGenerator()}), Globalize("en").pluralGenerator({}));
Globalize.b211172047 = messageFormatterFn((function( plural, pluralFuncs ) {
  return function(d) { return "“" + d.CONTENT_NAME + "” shared with " + plural(d.TOTAL_USER_COUNT, 0, pluralFuncs.en, { one: function() { return d.TOTAL_USER_COUNT_LOC + " user";}, other: function() { return d.TOTAL_USER_COUNT_LOC + " users";} }) + ". " + plural(d.UNPERMISSIONED_USER_COUNT, 0, pluralFuncs.en, { one: function() { return d.UNPERMISSIONED_USER_COUNT_LOC;}, other: function() { return d.UNPERMISSIONED_USER_COUNT_LOC;} }) + " of these users don't have permission to view the virtual connection. This virtual connection is in a project with [LOCKED_PERMISSIONS] and can't be managed individually."; }
})(messageFormat.plural, {"en": Globalize("en").pluralGenerator()}), Globalize("en").pluralGenerator({}));
Globalize.b832589272 = messageFormatterFn((function( plural, pluralFuncs ) {
  return function(d) { return "“" + d.CONTENT_NAME + "” shared with " + plural(d.TOTAL_USER_COUNT, 0, pluralFuncs.en, { one: function() { return d.TOTAL_USER_COUNT_LOC + " user";}, other: function() { return d.TOTAL_USER_COUNT_LOC + " users";} }) + ". " + plural(d.UNPERMISSIONED_USER_COUNT, 0, pluralFuncs.en, { one: function() { return d.UNPERMISSIONED_USER_COUNT_LOC;}, other: function() { return d.UNPERMISSIONED_USER_COUNT_LOC;} }) + " of these users don't have permission to view the workbook. This workbook is in a project with [LOCKED_PERMISSIONS] and can't be managed individually."; }
})(messageFormat.plural, {"en": Globalize("en").pluralGenerator()}), Globalize("en").pluralGenerator({}));
Globalize.a1078243736 = messageFormatterFn((function( plural, pluralFuncs ) {
  return function(d) { return "“" + d.CONTENT_NAME + "” shared with " + plural(d.TOTAL_USER_COUNT, 0, pluralFuncs.en, { one: function() { return d.TOTAL_USER_COUNT_LOC + " user";}, other: function() { return d.TOTAL_USER_COUNT_LOC + " users";} }) + ". " + plural(d.UNPERMISSIONED_USER_COUNT, 0, pluralFuncs.en, { one: function() { return d.UNPERMISSIONED_USER_COUNT_LOC;}, other: function() { return d.UNPERMISSIONED_USER_COUNT_LOC;} }) + " of these users don't have permission to view the view. This view is in a workbook with [TABBED_VIEWS] and permissions can't be managed individually."; }
})(messageFormat.plural, {"en": Globalize("en").pluralGenerator()}), Globalize("en").pluralGenerator({}));
Globalize.a598375148 = messageFormatterFn((function(  ) {
  return function(d) { return "“" + d.CONTENT_NAME + "” shared with one user. This user doesn't have permission to view the collection. Do you want to grant this user view access?"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1971220179 = messageFormatterFn((function(  ) {
  return function(d) { return "“" + d.CONTENT_NAME + "” shared with one user. This user doesn't have permission to view the datasource. Do you want to grant this user the view permissions template?"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a716121020 = messageFormatterFn((function(  ) {
  return function(d) { return "“" + d.CONTENT_NAME + "” shared with one user. This user doesn't have permission to view the flow. Do you want to grant this user the view permissions template?"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1464604930 = messageFormatterFn((function(  ) {
  return function(d) { return "“" + d.CONTENT_NAME + "” shared with one user. This user doesn't have permission to view the metric. Do you want to grant this user the view permissions template?"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1520987873 = messageFormatterFn((function(  ) {
  return function(d) { return "“" + d.CONTENT_NAME + "” shared with one user. This user doesn't have permission to view the project. Do you want to grant this user the view permissions template?"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1171118963 = messageFormatterFn((function(  ) {
  return function(d) { return "“" + d.CONTENT_NAME + "” shared with one user. This user doesn't have permission to view the view. Do you want to grant this user the view permissions template?"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1163447343 = messageFormatterFn((function(  ) {
  return function(d) { return "“" + d.CONTENT_NAME + "” shared with one user. This user doesn't have permission to view the virtual connection. Do you want to grant this user the view permissions template?"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1403867016 = messageFormatterFn((function(  ) {
  return function(d) { return "“" + d.CONTENT_NAME + "” shared with one user. This user doesn't have permission to view the workbook. Do you want to grant this user the view permissions template?"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b395522269 = messageFormatterFn((function(  ) {
  return function(d) { return "“" + d.CONTENT_NAME + "” shared with one user. This user doesn't have permission to view the datasource. This datasource is in a project with [LOCKED_PERMISSIONS] and can't be managed individually."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2034237428 = messageFormatterFn((function(  ) {
  return function(d) { return "“" + d.CONTENT_NAME + "” shared with one user. This user doesn't have permission to view the flow. This flow is in a project with [LOCKED_PERMISSIONS] and can't be managed individually."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1140780878 = messageFormatterFn((function(  ) {
  return function(d) { return "“" + d.CONTENT_NAME + "” shared with one user. This user doesn't have permission to view the metric. This metric is in a project with [LOCKED_PERMISSIONS] and can't be managed individually."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a683569297 = messageFormatterFn((function(  ) {
  return function(d) { return "“" + d.CONTENT_NAME + "” shared with one user. This user doesn't have permission to view the project. This project is in a project with [LOCKED_PERMISSIONS] and can't be managed individually."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1579239485 = messageFormatterFn((function(  ) {
  return function(d) { return "“" + d.CONTENT_NAME + "” shared with one user. This user doesn't have permission to view the view. This view is in a project with [LOCKED_PERMISSIONS] and can't be managed individually."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1490364031 = messageFormatterFn((function(  ) {
  return function(d) { return "“" + d.CONTENT_NAME + "” shared with one user. This user doesn't have permission to view the virtual connection. This virtual connection is in a project with [LOCKED_PERMISSIONS] and can't be managed individually."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1213694936 = messageFormatterFn((function(  ) {
  return function(d) { return "“" + d.CONTENT_NAME + "” shared with one user. This user doesn't have permission to view the workbook. This workbook is in a project with [LOCKED_PERMISSIONS] and can't be managed individually."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1078312424 = messageFormatterFn((function(  ) {
  return function(d) { return "“" + d.CONTENT_NAME + "” shared with one user. This user doesn't have permission to view the view. This workbook is in a workbook with [TABBED_VIEWS] and permissions can't be managed individually."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1226358424 = messageFormatterFn((function(  ) {
  return function(d) { return "Granting the view permissions template will give users the capability to view and connect to a datasource."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a360368657 = messageFormatterFn((function(  ) {
  return function(d) { return "The view template gives users the capability to view a flow."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a254757907 = messageFormatterFn((function(  ) {
  return function(d) { return "The view template gives users the capability to view a metric."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1013339028 = messageFormatterFn((function(  ) {
  return function(d) { return "The view template gives users the capability to view a project."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a815366600 = messageFormatterFn((function(  ) {
  return function(d) { return "Granting the view permissions template will give users the capability to view, filter, comment, and download an image, PDF, or summary data."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1055047780 = messageFormatterFn((function(  ) {
  return function(d) { return "Granting the view permissions template will give users the capability to view and connect to a virtual connection."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a149144413 = messageFormatterFn((function(  ) {
  return function(d) { return "Granting the view permissions template will give users the capability to view, filter, comment, and download an image, PDF, or summary data."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b842290869 = messageFormatterFn((function(  ) {
  return function(d) { return "Grant Access to Collection"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a530554162 = messageFormatterFn((function(  ) {
  return function(d) { return "Grant Access to Datasource"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a393102683 = messageFormatterFn((function(  ) {
  return function(d) { return "Grant Access to Flow"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1647385821 = messageFormatterFn((function(  ) {
  return function(d) { return "Grant Access to Metric"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b791546654 = messageFormatterFn((function(  ) {
  return function(d) { return "Grant Access to Project"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a848100626 = messageFormatterFn((function(  ) {
  return function(d) { return "Grant Access to View"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1677333358 = messageFormatterFn((function(  ) {
  return function(d) { return "Grant Access to Virtual Connection"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1565226585 = messageFormatterFn((function(  ) {
  return function(d) { return "Grant Access to Workbook"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1758419554 = messageFormatterFn((function(  ) {
  return function(d) { return "Unable to share \"" + d.CONTENT_NAME + "\"."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1012149183 = messageFormatterFn((function(  ) {
  return function(d) { return "\"" + d.CONTENT_NAME + "\" shared."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1192907857 = messageFormatterFn((function(  ) {
  return function(d) { return "Only people with permission can see this " + d.CONTENT_TYPE + "."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1322667946 = messageFormatterFn((function(  ) {
  return function(d) { return "Share the current visualization. Only people with permission can view this."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1101050456 = messageFormatterFn((function(  ) {
  return function(d) { return "Share this explanation and the accompanying view. Only people with permission can view this. If the data changes after sharing, explanations might change."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2030374381 = messageFormatterFn((function(  ) {
  return function(d) { return "In a collection, people see only the items they have permission to access."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1535026880 = messageFormatterFn((function(  ) {
  return function(d) { return "Message (optional)"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a795588134 = messageFormatterFn((function(  ) {
  return function(d) { return "Site ID copied to the clipboard."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1879483842 = messageFormatterFn((function(  ) {
  return function(d) { return "Site ID"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1802721148 = messageFormatterFn((function(  ) {
  return function(d) { return "view"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1563241497 = messageFormatterFn((function(  ) {
  return function(d) { return "workbook"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b721942866 = messageFormatterFn((function( plural, pluralFuncs ) {
  return function(d) { return plural(d.fieldCount, 0, pluralFuncs.en, { one: function() { return "Unrelated dimension: ";}, other: function() { return "Unrelated dimensions: ";} }); }
})(messageFormat.plural, {"en": Globalize("en").pluralGenerator()}), Globalize("en").pluralGenerator({}));
Globalize.a1658281270 = messageFormatterFn((function( plural, pluralFuncs ) {
  return function(d) { return plural(d.fieldCount, 0, pluralFuncs.en, { one: function() { return "Unrelated measure: ";}, other: function() { return "Unrelated measures: ";} }); }
})(messageFormat.plural, {"en": Globalize("en").pluralGenerator()}), Globalize("en").pluralGenerator({}));
Globalize.a1251770598 = messageFormatterFn((function( plural, pluralFuncs ) {
  return function(d) { return plural(d.fieldCount, 0, pluralFuncs.en, { one: function() { return "Ambiguously related dimension: ";}, other: function() { return "Ambiguously related dimensions: ";} }); }
})(messageFormat.plural, {"en": Globalize("en").pluralGenerator()}), Globalize("en").pluralGenerator({}));
Globalize.b1119550098 = messageFormatterFn((function( plural, pluralFuncs ) {
  return function(d) { return plural(d.fieldCount, 0, pluralFuncs.en, { one: function() { return "Ambiguously related measure: ";}, other: function() { return "Ambiguously related measures: ";} }); }
})(messageFormat.plural, {"en": Globalize("en").pluralGenerator()}), Globalize("en").pluralGenerator({}));
Globalize.b2062073146 = messageFormatterFn((function(  ) {
  return function(d) { return "Apply Filter to Worksheets " + d.fieldCaption; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b106205028 = messageFormatterFn((function(  ) {
  return function(d) { return "OK"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b240011462 = messageFormatterFn((function(  ) {
  return function(d) { return "Cancel"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a989781303 = messageFormatterFn((function(  ) {
  return function(d) { return "All"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1912111949 = messageFormatterFn((function(  ) {
  return function(d) { return "Select all on dashboard"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a993521080 = messageFormatterFn((function(  ) {
  return function(d) { return "None"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1183767749 = messageFormatterFn((function(  ) {
  return function(d) { return "Worksheets"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1778311240 = messageFormatterFn((function(  ) {
  return function(d) { return "Relationship"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1300001768 = messageFormatterFn((function(  ) {
  return function(d) { return "Details"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a691596856 = messageFormatterFn((function(  ) {
  return function(d) { return "Show only selected worksheets"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b151413729 = messageFormatterFn((function(  ) {
  return function(d) { return "Show all worksheets in workbook"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a901628261 = messageFormatterFn((function(  ) {
  return function(d) { return "Filter will be applied to 1 worksheet."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b765693779 = messageFormatterFn((function(  ) {
  return function(d) { return "Filter will be applied to " + d.count + " worksheets."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b467862399 = messageFormatterFn((function(  ) {
  return function(d) { return "Enter search text"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a858861844 = messageFormatterFn((function(  ) {
  return function(d) { return "Select all worksheets"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a740207144 = messageFormatterFn((function(  ) {
  return function(d) { return "Select " + d.worksheetName + ". Checkbox unchecked"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1316895481 = messageFormatterFn((function(  ) {
  return function(d) { return "Deselect " + d.worksheetName + ". Checkbox checked"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1077852668 = messageFormatterFn((function(  ) {
  return function(d) { return "Filter By"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b443006251 = messageFormatterFn((function(  ) {
  return function(d) { return "Sort By"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a434622246 = messageFormatterFn((function(  ) {
  return function(d) { return "Alphabetic"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2118864080 = messageFormatterFn((function(  ) {
  return function(d) { return "Data source order"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b952501343 = messageFormatterFn((function(  ) {
  return function(d) { return "Field"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a603830175 = messageFormatterFn((function(  ) {
  return function(d) { return "Manual"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2139263024 = messageFormatterFn((function(  ) {
  return function(d) { return "Nested"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1476047259 = messageFormatterFn((function(  ) {
  return function(d) { return "Sort Order"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1088821292 = messageFormatterFn((function(  ) {
  return function(d) { return "Ascending"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1383025708 = messageFormatterFn((function(  ) {
  return function(d) { return "Descending"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1590542138 = messageFormatterFn((function(  ) {
  return function(d) { return "Aggregation"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1655818490 = messageFormatterFn((function(  ) {
  return function(d) { return "Clear"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1410814403 = messageFormatterFn((function(  ) {
  return function(d) { return "Field Name"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a787778813 = messageFormatterFn((function(  ) {
  return function(d) { return "Move up"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b599368426 = messageFormatterFn((function(  ) {
  return function(d) { return "Move down"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b955194377 = messageFormatterFn((function(  ) {
  return function(d) { return "Move to top"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b727685727 = messageFormatterFn((function(  ) {
  return function(d) { return "Move to bottom"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a131510542 = messageFormatterFn((function(  ) {
  return function(d) { return "OK"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1261262676 = messageFormatterFn((function(  ) {
  return function(d) { return "Cancel"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b202826446 = messageFormatterFn((function(  ) {
  return function(d) { return "True"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2027329707 = messageFormatterFn((function(  ) {
  return function(d) { return "False"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b595442054 = messageFormatterFn((function(  ) {
  return function(d) { return ","; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1278458837 = messageFormatterFn((function(  ) {
  return function(d) { return "Parameter"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1300056718 = messageFormatterFn((function(  ) {
  return function(d) { return "Type"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1563161741 = messageFormatterFn((function(  ) {
  return function(d) { return "Value"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2085797674 = messageFormatterFn((function(  ) {
  return function(d) { return "Sign in to reconnect"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1439999088 = messageFormatterFn((function(  ) {
  return function(d) { return "The username or password is not valid.  Check the database name and credentials and try again."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1833479746 = messageFormatterFn((function(  ) {
  return function(d) { return "Copy"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2019245423 = messageFormatterFn((function(  ) {
  return function(d) { return "Cut"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1546192128 = messageFormatterFn((function(  ) {
  return function(d) { return "Paste"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a594888614 = messageFormatterFn((function(  ) {
  return function(d) { return "Remember Password"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1611623643 = messageFormatterFn((function(  ) {
  return function(d) { return "Detailed Error Message"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1133156375 = messageFormatterFn((function(  ) {
  return function(d) { return "Optional"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a708277281 = messageFormatterFn((function(  ) {
  return function(d) { return "Sign In"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a805792974 = messageFormatterFn((function(  ) {
  return function(d) { return "Signing In…"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a934531862 = messageFormatterFn((function(  ) {
  return function(d) { return "Username:"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1120794191 = messageFormatterFn((function(  ) {
  return function(d) { return "Password:"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1980010255 = messageFormatterFn((function(  ) {
  return function(d) { return "Hide connection details"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b208402060 = messageFormatterFn((function(  ) {
  return function(d) { return "Show connection details"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1838164408 = messageFormatterFn((function(  ) {
  return function(d) { return "Download and install the drivers,"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a290672348 = messageFormatterFn((function(  ) {
  return function(d) { return " and then connect."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a838543130 = messageFormatterFn((function(  ) {
  return function(d) { return "No driver installed for " + d.connectorName + " connector. Contact your administrator."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a571643587 = messageFormatterFn((function(  ) {
  return function(d) { return "Show template project contributors!"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b37372856 = messageFormatterFn((function(  ) {
  return function(d) { return "Hide template project contributors!"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1432712825 = messageFormatterFn((function(  ) {
  return function(d) { return "Field separator"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1852282661 = messageFormatterFn((function(  ) {
  return function(d) { return "Text qualifier"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a576444500 = messageFormatterFn((function(  ) {
  return function(d) { return "Character set"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a622425352 = messageFormatterFn((function(  ) {
  return function(d) { return "Locale"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1524115027 = messageFormatterFn((function(  ) {
  return function(d) { return "Automatic"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a156559142 = messageFormatterFn((function(  ) {
  return function(d) { return "None"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a717543662 = messageFormatterFn((function(  ) {
  return function(d) { return "Space"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a457557420 = messageFormatterFn((function(  ) {
  return function(d) { return "No match"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1583549560 = messageFormatterFn((function(  ) {
  return function(d) { return "Other"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1444330029 = messageFormatterFn((function(  ) {
  return function(d) { return "search"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b614866179 = messageFormatterFn((function(  ) {
  return function(d) { return "Comma"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b959670531 = messageFormatterFn((function(  ) {
  return function(d) { return "Tab"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a396836697 = messageFormatterFn((function(  ) {
  return function(d) { return "Semicolon"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1270321445 = messageFormatterFn((function(  ) {
  return function(d) { return "Vertical Bar"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1579049296 = messageFormatterFn((function(  ) {
  return function(d) { return "Trend Lines Options"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1868379190 = messageFormatterFn((function(  ) {
  return function(d) { return "Build separate trend lines based on the following dimensions:"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1947347274 = messageFormatterFn((function(  ) {
  return function(d) { return "Model Type"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a819849079 = messageFormatterFn((function(  ) {
  return function(d) { return "Factors"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b302367087 = messageFormatterFn((function(  ) {
  return function(d) { return "Options"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b687987560 = messageFormatterFn((function(  ) {
  return function(d) { return "Exponential"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a76528456 = messageFormatterFn((function(  ) {
  return function(d) { return "Linear"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1727818734 = messageFormatterFn((function(  ) {
  return function(d) { return "Logarithmic"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b423836072 = messageFormatterFn((function(  ) {
  return function(d) { return "Power"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b441958601 = messageFormatterFn((function(  ) {
  return function(d) { return "Polynomial"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a923177507 = messageFormatterFn((function(  ) {
  return function(d) { return "Degree:"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1370509998 = messageFormatterFn((function(  ) {
  return function(d) { return "OK"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1524134610 = messageFormatterFn((function(  ) {
  return function(d) { return "Allow a trend line per color"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b198180353 = messageFormatterFn((function(  ) {
  return function(d) { return "Show confidence bands"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b253066800 = messageFormatterFn((function(  ) {
  return function(d) { return "Show recalculated line for highlighted or selected data points"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a248783163 = messageFormatterFn((function(  ) {
  return function(d) { return "Show tooltips"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a331772532 = messageFormatterFn((function(  ) {
  return function(d) { return "Force y-intercept to zero"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b799949357 = messageFormatterFn((function(  ) {
  return function(d) { return "Drag table to union"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1997338155 = messageFormatterFn((function(  ) {
  return function(d) { return "Apply"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b485188897 = messageFormatterFn((function(  ) {
  return function(d) { return "OK"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b495683726 = messageFormatterFn((function(  ) {
  return function(d) { return "Connection: "; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b372846168 = messageFormatterFn((function(  ) {
  return function(d) { return "Drag tables here"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1771158506 = messageFormatterFn((function(  ) {
  return function(d) { return "Tables in union: "; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a458193300 = messageFormatterFn((function(  ) {
  return function(d) { return "You can only union data from the same connection"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1231534210 = messageFormatterFn((function(  ) {
  return function(d) { return "Tableau can't find or recognize this table. Remove or replace the table"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a836152981 = messageFormatterFn((function(  ) {
  return function(d) { return "Only the first table in your selection will be added because union is not available for this database."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a117205912 = messageFormatterFn((function(  ) {
  return function(d) { return "Only the first table in your selection will be added because union is not available for this database."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1859531150 = messageFormatterFn((function(  ) {
  return function(d) { return "Search In: "; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1636481881 = messageFormatterFn((function(  ) {
  return function(d) { return "Sheets"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b98481494 = messageFormatterFn((function(  ) {
  return function(d) { return "Workbooks"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b601554992 = messageFormatterFn((function(  ) {
  return function(d) { return "Matching pattern (xxx*)"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1979813371 = messageFormatterFn((function(  ) {
  return function(d) { return "blank = include all"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1068017801 = messageFormatterFn((function(  ) {
  return function(d) { return "blank = exclude none"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1779454130 = messageFormatterFn((function(  ) {
  return function(d) { return "Expand search to subfolders"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2117788827 = messageFormatterFn((function(  ) {
  return function(d) { return "Expand search to parent folder"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1101528362 = messageFormatterFn((function(  ) {
  return function(d) { return "Learn more"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1908602095 = messageFormatterFn((function(  ) {
  return function(d) { return "Include"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a796719521 = messageFormatterFn((function(  ) {
  return function(d) { return "Exclude"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a591208836 = messageFormatterFn((function(  ) {
  return function(d) { return "Apply (Ctrl + Enter)"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b485387594 = messageFormatterFn((function(  ) {
  return function(d) { return "Apply and Close (Enter)"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b91217801 = messageFormatterFn((function(  ) {
  return function(d) { return "Use * to match any string of characters."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1894825808 = messageFormatterFn((function(  ) {
  return function(d) { return "For example, "; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b619738354 = messageFormatterFn((function(  ) {
  return function(d) { return "Sales* "; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b733421407 = messageFormatterFn((function(  ) {
  return function(d) { return "finds "; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a950643183 = messageFormatterFn((function(  ) {
  return function(d) { return "Sales.csv "; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a481906782 = messageFormatterFn((function(  ) {
  return function(d) { return "and "; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a781009207 = messageFormatterFn((function(  ) {
  return function(d) { return "Sales 1999.csv."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a582441191 = messageFormatterFn((function(  ) {
  return function(d) { return "Specific (manual)"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b329701569 = messageFormatterFn((function(  ) {
  return function(d) { return "Wildcard (automatic)"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1245424851 = messageFormatterFn((function(  ) {
  return function(d) { return "Remove " + d.PILL_NAME; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b603815914 = messageFormatterFn((function(  ) {
  return function(d) { return "Cancel"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1212669750 = messageFormatterFn((function(  ) {
  return function(d) { return "OK"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a395161007 = messageFormatterFn((function(  ) {
  return function(d) { return "No"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1799562621 = messageFormatterFn((function(  ) {
  return function(d) { return "Yes"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1830185684 = messageFormatterFn((function(  ) {
  return function(d) { return "Do you want to turn off the auto-generated layout and create a custom one?"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1195625754 = messageFormatterFn((function(  ) {
  return function(d) { return "Edit Layout"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a835513770 = messageFormatterFn((function(  ) {
  return function(d) { return "Don't show this again"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a443105541 = messageFormatterFn((function(  ) {
  return function(d) { return "(All)"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a202851998 = messageFormatterFn((function(  ) {
  return function(d) { return "Close View Data to see download information."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b682478214 = messageFormatterFn((function(  ) {
  return function(d) { return "Download"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a10497558 = messageFormatterFn((function(  ) {
  return function(d) { return "No column data available to download"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a809885376 = messageFormatterFn((function(  ) {
  return function(d) { return "See the Tableau browser window for download information."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1569608633 = messageFormatterFn((function(  ) {
  return function(d) { return "Expand tab pane"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b750978378 = messageFormatterFn((function( plural, pluralFuncs ) {
  return function(d) { return plural(d.fieldCount, 0, pluralFuncs.en, { one: function() { return "Move column to first";}, other: function() { return "Move columns to first";} }); }
})(messageFormat.plural, {"en": Globalize("en").pluralGenerator()}), Globalize("en").pluralGenerator({}));
Globalize.a1054554365 = messageFormatterFn((function( plural, pluralFuncs ) {
  return function(d) { return plural(d.fieldCount, 0, pluralFuncs.en, { one: function() { return "Move column to left";}, other: function() { return "Move columns to left";} }); }
})(messageFormat.plural, {"en": Globalize("en").pluralGenerator()}), Globalize("en").pluralGenerator({}));
Globalize.b1065322856 = messageFormatterFn((function( plural, pluralFuncs ) {
  return function(d) { return plural(d.fieldCount, 0, pluralFuncs.en, { one: function() { return "Move column to right";}, other: function() { return "Move columns to right";} }); }
})(messageFormat.plural, {"en": Globalize("en").pluralGenerator()}), Globalize("en").pluralGenerator({}));
Globalize.b1202263041 = messageFormatterFn((function( plural, pluralFuncs ) {
  return function(d) { return plural(d.fieldCount, 0, pluralFuncs.en, { one: function() { return "Move column to last";}, other: function() { return "Move columns to last";} }); }
})(messageFormat.plural, {"en": Globalize("en").pluralGenerator()}), Globalize("en").pluralGenerator({}));
Globalize.b1276021374 = messageFormatterFn((function( plural, pluralFuncs ) {
  return function(d) { return plural(d.fieldCount, 0, pluralFuncs.en, { one: function() { return "Make " + d.fieldName + " column the first column in the table";}, other: function() { return "Make selected columns the first columns in the table";} }); }
})(messageFormat.plural, {"en": Globalize("en").pluralGenerator()}), Globalize("en").pluralGenerator({}));
Globalize.b121071223 = messageFormatterFn((function( plural, pluralFuncs ) {
  return function(d) { return plural(d.fieldCount, 0, pluralFuncs.en, { one: function() { return "Move " + d.fieldName + " column one position to the left";}, other: function() { return "Move selected columns one position to the left";} }); }
})(messageFormat.plural, {"en": Globalize("en").pluralGenerator()}), Globalize("en").pluralGenerator({}));
Globalize.b498695708 = messageFormatterFn((function( plural, pluralFuncs ) {
  return function(d) { return plural(d.fieldCount, 0, pluralFuncs.en, { one: function() { return "Move " + d.fieldName + " column one position to the right";}, other: function() { return "Move selected columns one position to the right";} }); }
})(messageFormat.plural, {"en": Globalize("en").pluralGenerator()}), Globalize("en").pluralGenerator({}));
Globalize.b1942412021 = messageFormatterFn((function( plural, pluralFuncs ) {
  return function(d) { return plural(d.fieldCount, 0, pluralFuncs.en, { one: function() { return "Make " + d.fieldName + " column the last column in the table";}, other: function() { return "Make selected columns the last columns in the table";} }); }
})(messageFormat.plural, {"en": Globalize("en").pluralGenerator()}), Globalize("en").pluralGenerator({}));
Globalize.b221745046 = messageFormatterFn((function(  ) {
  return function(d) { return "Move Left"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1433914641 = messageFormatterFn((function(  ) {
  return function(d) { return "Move Right"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1035039786 = messageFormatterFn((function(  ) {
  return function(d) { return "Move to First"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b876247805 = messageFormatterFn((function(  ) {
  return function(d) { return "Move to Last"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b131299274 = messageFormatterFn((function(  ) {
  return function(d) { return "Remove"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a467179710 = messageFormatterFn((function(  ) {
  return function(d) { return "Sort Ascending"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b819653784 = messageFormatterFn((function(  ) {
  return function(d) { return "Sort Descending"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b60890193 = messageFormatterFn((function(  ) {
  return function(d) { return "Clear Sort"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1651133349 = messageFormatterFn((function(  ) {
  return function(d) { return "Show Aliases"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1620094008 = messageFormatterFn((function(  ) {
  return function(d) { return "Show Fields"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b298819291 = messageFormatterFn((function(  ) {
  return function(d) { return "Search fields"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a286150577 = messageFormatterFn((function(  ) {
  return function(d) { return "Select fields to display"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1569435221 = messageFormatterFn((function(  ) {
  return function(d) { return "Settings"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b513877545 = messageFormatterFn((function(  ) {
  return function(d) { return "Tables"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1384943216 = messageFormatterFn((function(  ) {
  return function(d) { return "Tabs"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1341239819 = messageFormatterFn((function(  ) {
  return function(d) { return "rows"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1792448229 = messageFormatterFn((function(  ) {
  return function(d) { return "Update row limit button"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a56913092 = messageFormatterFn((function(  ) {
  return function(d) { return "Enter the maximum number of rows to display, then press the Enter key"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1034549058 = messageFormatterFn((function(  ) {
  return function(d) { return "View Data table"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a940445531 = messageFormatterFn((function(  ) {
  return function(d) { return "Tab options for what will be displayed in the View Data table"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1645884756 = messageFormatterFn((function(  ) {
  return function(d) { return d.tableName + " field Table"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b457552600 = messageFormatterFn((function(  ) {
  return function(d) { return d.tableName + " Table"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1091873993 = messageFormatterFn((function(  ) {
  return function(d) { return d.tableName + " custom SQL Table"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b747934899 = messageFormatterFn((function(  ) {
  return function(d) { return "Collapse navigation pane"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1117640378 = messageFormatterFn((function(  ) {
  return function(d) { return "Expand navigation pane"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1716998876 = messageFormatterFn((function( plural, pluralFuncs ) {
  return function(d) { return plural(d.rowCount, 0, pluralFuncs.en, { one: function() { return d.formattedRowCount + " row";}, other: function() { return d.formattedRowCount + " rows";} }) + "  " + plural(d.fieldCount, 0, pluralFuncs.en, { one: function() { return d.formattedFieldCount + " field";}, other: function() { return d.formattedFieldCount + " fields";} }); }
})(messageFormat.plural, {"en": Globalize("en").pluralGenerator()}), Globalize("en").pluralGenerator({}));
Globalize.a1893224290 = messageFormatterFn((function( plural, pluralFuncs ) {
  return function(d) { return plural(d.fieldCount, 0, pluralFuncs.en, { one: function() { return d.formattedFieldCount + " field";}, other: function() { return d.formattedFieldCount + " fields";} }) + " selected"; }
})(messageFormat.plural, {"en": Globalize("en").pluralGenerator()}), Globalize("en").pluralGenerator({}));
Globalize.a486084011 = messageFormatterFn((function( plural, pluralFuncs ) {
  return function(d) { return plural(d.fieldCount, 0, pluralFuncs.en, { one: function() { return d.fieldName + " was removed";}, other: function() { return "Columns were removed";} }); }
})(messageFormat.plural, {"en": Globalize("en").pluralGenerator()}), Globalize("en").pluralGenerator({}));
Globalize.a23240513 = messageFormatterFn((function(  ) {
  return function(d) { return "Rows count updated to " + d.value; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2119973208 = messageFormatterFn((function(  ) {
  return function(d) { return "Rows count input " + d.value + " was invalid"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a595022570 = messageFormatterFn((function(  ) {
  return function(d) { return d.fieldName + " is now in position " + d.columnPosition; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1095722342 = messageFormatterFn((function(  ) {
  return function(d) { return "Show template project contributors!"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a486705899 = messageFormatterFn((function(  ) {
  return function(d) { return "Hide template project contributors!"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1095749133 = messageFormatterFn((function(  ) {
  return function(d) { return "Hide Unused Tables"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1481764647 = messageFormatterFn((function(  ) {
  return function(d) { return "Show Fields"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a656779206 = messageFormatterFn((function(  ) {
  return function(d) { return "View Data Model"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b512005133 = messageFormatterFn((function(  ) {
  return function(d) { return "View Data Model is for worksheets connected to a single, non-cube data source. Switch to a worksheet that doesn't use a cube or data blending to use this feature."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a809050468 = messageFormatterFn((function( plural, pluralFuncs ) {
  return function(d) { return d.failureCountString + " " + plural(d.failureCount, 0, pluralFuncs.en, { one: function() { return "Alert";}, other: function() { return "Alerts";} }); }
})(messageFormat.plural, {"en": Globalize("en").pluralGenerator()}), Globalize("en").pluralGenerator({}));
Globalize.b1911266762 = messageFormatterFn((function(  ) {
  return function(d) { return "Validation failure"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a245120581 = messageFormatterFn((function(  ) {
  return function(d) { return "Accessibility"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b391927113 = messageFormatterFn((function(  ) {
  return function(d) { return "Alternative text (alt text) is read aloud to people who use screen reading software. Tableau generates alt text automatically, and you can customize the alt text for your viz."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1011724011 = messageFormatterFn((function(  ) {
  return function(d) { return "Alt text description"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b334942471 = messageFormatterFn((function(  ) {
  return function(d) { return "Enter descriptive alt text"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1957585546 = messageFormatterFn((function(  ) {
  return function(d) { return "Reset"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a517236525 = messageFormatterFn((function(  ) {
  return function(d) { return "Cancel"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1751476111 = messageFormatterFn((function(  ) {
  return function(d) { return "OK"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1710519335 = messageFormatterFn((function(  ) {
  return function(d) { return "Data"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2017707608 = messageFormatterFn((function(  ) {
  return function(d) { return "Discover"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1921417719 = messageFormatterFn((function(  ) {
  return function(d) { return "Display"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1967999568 = messageFormatterFn((function(  ) {
  return function(d) { return "Change display settings"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b775656649 = messageFormatterFn((function(  ) {
  return function(d) { return "Download"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b893137566 = messageFormatterFn((function(  ) {
  return function(d) { return "Choose a format to download"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1679429063 = messageFormatterFn((function(  ) {
  return function(d) { return "Edit"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1376596752 = messageFormatterFn((function(  ) {
  return function(d) { return "More options..."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2036261311 = messageFormatterFn((function(  ) {
  return function(d) { return "Visualization controls"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1803665962 = messageFormatterFn((function(  ) {
  return function(d) { return "Watch"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1905564131 = messageFormatterFn((function(  ) {
  return function(d) { return "Keep track of updates"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1190454750 = messageFormatterFn((function(  ) {
  return function(d) { return "Cancel"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a721395064 = messageFormatterFn((function(  ) {
  return function(d) { return "pending"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b937005201 = messageFormatterFn((function(  ) {
  return function(d) { return "in progress"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b247787745 = messageFormatterFn((function(  ) {
  return function(d) { return "completed"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1846015039 = messageFormatterFn((function(  ) {
  return function(d) { return d.headerText + " with " + d.stepCount + " steps"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2042505372 = messageFormatterFn((function(  ) {
  return function(d) { return "step " + d.stepNumber + " is done, "; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1121211749 = messageFormatterFn((function(  ) {
  return function(d) { return "starting step " + d.stepNumber; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2052355194 = messageFormatterFn((function(  ) {
  return function(d) { return "Add a tag"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1773237779 = messageFormatterFn((function(  ) {
  return function(d) { return "Save"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a930426128 = messageFormatterFn((function(  ) {
  return function(d) { return "Cancel"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1900327897 = messageFormatterFn((function(  ) {
  return function(d) { return "Show template project contributors!"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1785622956 = messageFormatterFn((function(  ) {
  return function(d) { return "Hide template project contributors!"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a481411373 = messageFormatterFn((function(  ) {
  return function(d) { return "Edit URL"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b772312104 = messageFormatterFn((function(  ) {
  return function(d) { return "OK"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1760884495 = messageFormatterFn((function(  ) {
  return function(d) { return "URL"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1240648174 = messageFormatterFn((function(  ) {
  return function(d) { return "Enter the URL"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a270597722 = messageFormatterFn((function(  ) {
  return function(d) { return "Insert"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a6571815 = messageFormatterFn((function(  ) {
  return function(d) { return "Alerts"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a965676245 = messageFormatterFn((function(  ) {
  return function(d) { return "Bold"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1732024758 = messageFormatterFn((function(  ) {
  return function(d) { return "Cancel"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1583464650 = messageFormatterFn((function(  ) {
  return function(d) { return d.COUNT + " of " + d.MAXIMUM + " characters used"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a676088830 = messageFormatterFn((function(  ) {
  return function(d) { return "Close"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1532506388 = messageFormatterFn((function(  ) {
  return function(d) { return "Close alert"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1586131674 = messageFormatterFn((function(  ) {
  return function(d) { return "Close dialog"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b598114794 = messageFormatterFn((function(  ) {
  return function(d) { return ": "; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1124383574 = messageFormatterFn((function(  ) {
  return function(d) { return "Dark Color Palette"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1191722766 = messageFormatterFn((function(  ) {
  return function(d) { return "Flow Color Palette"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b774890548 = messageFormatterFn((function(  ) {
  return function(d) { return "Light Color Palette"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b637250120 = messageFormatterFn((function(  ) {
  return function(d) { return "Custom color"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1041330970 = messageFormatterFn((function(  ) {
  return function(d) { return "Edit"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1734418304 = messageFormatterFn((function(  ) {
  return function(d) { return "Editor Mode"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b426049886 = messageFormatterFn((function(  ) {
  return function(d) { return "Enter a valid date."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b295924671 = messageFormatterFn((function(  ) {
  return function(d) { return "Error: "; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1750746339 = messageFormatterFn((function(  ) {
  return function(d) { return "Formatting Guide"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a724029266 = messageFormatterFn((function(  ) {
  return function(d) { return "Bold"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a987240285 = messageFormatterFn((function(  ) {
  return function(d) { return "To get this"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a697015723 = messageFormatterFn((function(  ) {
  return function(d) { return "Type this"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1474603068 = messageFormatterFn((function(  ) {
  return function(d) { return "Image"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b290771668 = messageFormatterFn((function(  ) {
  return function(d) { return "Italics"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1004845959 = messageFormatterFn((function(  ) {
  return function(d) { return "Link"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a37934581 = messageFormatterFn((function(  ) {
  return function(d) { return "Underline"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2121715777 = messageFormatterFn((function(  ) {
  return function(d) { return "Heading 1"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2121714816 = messageFormatterFn((function(  ) {
  return function(d) { return "Heading 2"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2121713855 = messageFormatterFn((function(  ) {
  return function(d) { return "Heading 3"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1541400977 = messageFormatterFn((function(  ) {
  return function(d) { return "Info: "; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a249191625 = messageFormatterFn((function(  ) {
  return function(d) { return "Italics"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a963803833 = messageFormatterFn((function(  ) {
  return function(d) { return "Ctrl+"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a776341578 = messageFormatterFn((function(  ) {
  return function(d) { return "Shift+"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1231038590 = messageFormatterFn((function(  ) {
  return function(d) { return "Loading…"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1246492938 = messageFormatterFn((function(  ) {
  return function(d) { return "Link"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a942650263 = messageFormatterFn((function(  ) {
  return function(d) { return "link text"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a31704859 = messageFormatterFn((function(  ) {
  return function(d) { return "url"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b384041535 = messageFormatterFn((function(  ) {
  return function(d) { return "More navigation links"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1858013165 = messageFormatterFn((function(  ) {
  return function(d) { return "Expand navigation links"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a909784723 = messageFormatterFn((function(  ) {
  return function(d) { return "Next Month"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a505500216 = messageFormatterFn((function(  ) {
  return function(d) { return "No matches."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1458356336 = messageFormatterFn((function(  ) {
  return function(d) { return "(optional)"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1424406689 = messageFormatterFn((function(  ) {
  return function(d) { return d.COUNT + " options selected"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b84994127 = messageFormatterFn((function(  ) {
  return function(d) { return "Ordered List"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1017875340 = messageFormatterFn((function(  ) {
  return function(d) { return "Paragraph"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a832661262 = messageFormatterFn((function(  ) {
  return function(d) { return "Preview"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b248595313 = messageFormatterFn((function(  ) {
  return function(d) { return "Previous Month"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2121673260 = messageFormatterFn((function(  ) {
  return function(d) { return "Remove"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a234654959 = messageFormatterFn((function(  ) {
  return function(d) { return "(required)"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1188078362 = messageFormatterFn((function(  ) {
  return function(d) { return "Success: "; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1490724826 = messageFormatterFn((function(  ) {
  return function(d) { return "Waiting for content"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a141034459 = messageFormatterFn((function(  ) {
  return function(d) { return "Editing is disabled in preview mode"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1945656834 = messageFormatterFn((function(  ) {
  return function(d) { return "Textile markup guide"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1182798713 = messageFormatterFn((function(  ) {
  return function(d) { return "Textile markup preview"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a584030657 = messageFormatterFn((function(  ) {
  return function(d) { return "Information"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a798780447 = messageFormatterFn((function(  ) {
  return function(d) { return d.ITEM_NAME + ", In section: " + d.SECTION_NAME; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b748383662 = messageFormatterFn((function(  ) {
  return function(d) { return "Underline"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a60007160 = messageFormatterFn((function(  ) {
  return function(d) { return "Unordered List"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1041600694 = messageFormatterFn((function(  ) {
  return function(d) { return "Wait indicator"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1893256685 = messageFormatterFn((function(  ) {
  return function(d) { return "Warning: "; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a553908461 = messageFormatterFn((function(  ) {
  return function(d) { return "Week " + d.weekNum; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1870284551 = messageFormatterFn((function( plural, pluralFuncs ) {
  return function(d) { return plural(d.optionCount, 0, pluralFuncs.en, { one: function() { return d.formattedOptionCount + " option";}, other: function() { return d.formattedOptionCount + " options";} }) + " available."; }
})(messageFormat.plural, {"en": Globalize("en").pluralGenerator()}), Globalize("en").pluralGenerator({}));
Globalize.b1632348247 = messageFormatterFn((function(  ) {
  return function(d) { return "Column " + d.COLUMN_NAME + " resized to width " + d.COLUMN_WIDTH + " pixels."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1600754090 = messageFormatterFn((function(  ) {
  return function(d) { return "Grid Navigation enabled."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1596309411 = messageFormatterFn((function(  ) {
  return function(d) { return "Grid Navigation disabled."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a383308691 = messageFormatterFn((function(  ) {
  return function(d) { return "Select All Cells"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b111298067 = messageFormatterFn((function(  ) {
  return function(d) { return "Select Row"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1349447062 = messageFormatterFn((function(  ) {
  return function(d) { return "Optimize Workbook"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a347884167 = messageFormatterFn((function(  ) {
  return function(d) { return "All data sources"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b686112079 = messageFormatterFn((function(  ) {
  return function(d) { return "Laptop"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1953268723 = messageFormatterFn((function(  ) {
  return function(d) { return "Phone"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b37021635 = messageFormatterFn((function(  ) {
  return function(d) { return "Tablet"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1016822261 = messageFormatterFn((function(  ) {
  return function(d) { return "Building an extract with your data..."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1456136672 = messageFormatterFn((function(  ) {
  return function(d) { return "For sensitive data, change permissions on the published workbook to adjust access."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1389107356 = messageFormatterFn((function(  ) {
  return function(d) { return "If you're connecting to a lot of data, this could take some time. You'll receive an email when the dashboard is ready for you."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1279467625 = messageFormatterFn((function(  ) {
  return function(d) { return "Learn more"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a354889190 = messageFormatterFn((function(  ) {
  return function(d) { return "Creating workbook..."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1851628779 = messageFormatterFn((function(  ) {
  return function(d) { return "If you're connecting to a lot of data, this could take some time. You'll receive an email when the dashboard is ready for you."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1989638466 = messageFormatterFn((function(  ) {
  return function(d) { return "For use with"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1575499934 = messageFormatterFn((function(  ) {
  return function(d) { return "Available layouts"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2106951433 = messageFormatterFn((function(  ) {
  return function(d) { return "More like this"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1379445792 = messageFormatterFn((function(  ) {
  return function(d) { return "Cancel"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1734214183 = messageFormatterFn((function(  ) {
  return function(d) { return "Workbook name"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1676883838 = messageFormatterFn((function(  ) {
  return function(d) { return "Create Workbook"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1254004607 = messageFormatterFn((function(  ) {
  return function(d) { return "Select project"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1120231480 = messageFormatterFn((function(  ) {
  return function(d) { return "Create workbook"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1537425502 = messageFormatterFn((function(  ) {
  return function(d) { return "Use Dashboard"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a526459905 = messageFormatterFn((function(  ) {
  return function(d) { return "Sign in to " + d.CONNECTION_TYPE; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1384554378 = messageFormatterFn((function(  ) {
  return function(d) { return "Use Sample Data for " + d.TEMPLATE_NAME; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1385784428 = messageFormatterFn((function(  ) {
  return function(d) { return "Sign in to " + d.CONNECTION_TYPE + " for " + d.TEMPLATE_NAME; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b88805786 = messageFormatterFn((function(  ) {
  return function(d) { return "Preview " + d.TEMPLATE_NAME; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b962698757 = messageFormatterFn((function(  ) {
  return function(d) { return "Learn more"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b737053074 = messageFormatterFn((function(  ) {
  return function(d) { return "Start from one of our pre-built dashboards. All you have to do is sign in to your data source."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a738810947 = messageFormatterFn((function(  ) {
  return function(d) { return "Use Sample Data"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b794508294 = messageFormatterFn((function(  ) {
  return function(d) { return "Got it"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1371265143 = messageFormatterFn((function(  ) {
  return function(d) { return "Rename Dashboard Item"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1261588545 = messageFormatterFn((function(  ) {
  return function(d) { return "OK"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1649480957 = messageFormatterFn((function(  ) {
  return function(d) { return "Cancel"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b297062076 = messageFormatterFn((function(  ) {
  return function(d) { return "Default Name"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a459657241 = messageFormatterFn((function(  ) {
  return function(d) { return "Warning"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2063839779 = messageFormatterFn((function(  ) {
  return function(d) { return "Adding this field might yield " + d.totalCount + " unique values. The recommended maximum for this shelf is " + d.maxCount + "."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1527745472 = messageFormatterFn((function(  ) {
  return function(d) { return "Adding these fields might yield " + d.totalCount + " unique values. The recommended maximum for this shelf is " + d.maxCount + "."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1628155543 = messageFormatterFn((function(  ) {
  return function(d) { return "You can reduce the number of values by filtering this dimension."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1933954523 = messageFormatterFn((function(  ) {
  return function(d) { return "You can reduce the number of values by filtering dimensions."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1811072766 = messageFormatterFn((function(  ) {
  return function(d) { return "You can reduce the number of values by filtering dimensions or only using related dimensions in the view."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b846714736 = messageFormatterFn((function(  ) {
  return function(d) { return "Field"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1088464904 = messageFormatterFn((function(  ) {
  return function(d) { return "Unrelated"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b108940431 = messageFormatterFn((function(  ) {
  return function(d) { return "Unrelated dimensions show all possible value combinations"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a344497300 = messageFormatterFn((function(  ) {
  return function(d) { return "Members"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2044971365 = messageFormatterFn((function(  ) {
  return function(d) { return "Estimated unique values"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a647328684 = messageFormatterFn((function(  ) {
  return function(d) { return "Estimated number of unique values to be displayed: " + d.totalCount; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1547475329 = messageFormatterFn((function(  ) {
  return function(d) { return "Pause automatic updates"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b656481785 = messageFormatterFn((function(  ) {
  return function(d) { return "Cancel"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b358944730 = messageFormatterFn((function(  ) {
  return function(d) { return "Add field"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1483212390 = messageFormatterFn((function(  ) {
  return function(d) { return "Add all fields"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a436708591 = messageFormatterFn((function(  ) {
  return function(d) { return "Filter then add field"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1065152734 = messageFormatterFn((function(  ) {
  return function(d) { return "Learn more"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a486862571 = messageFormatterFn((function(  ) {
  return function(d) { return "Was this helpful?"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a342776537 = messageFormatterFn((function(  ) {
  return function(d) { return "Provide Additional Feedback"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b461970362 = messageFormatterFn((function(  ) {
  return function(d) { return "Helpful"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2136755149 = messageFormatterFn((function(  ) {
  return function(d) { return "Not helpful"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b363739722 = messageFormatterFn((function(  ) {
  return function(d) { return "Submit"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a300092701 = messageFormatterFn((function(  ) {
  return function(d) { return "Why wasn't it helpful?"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b681930984 = messageFormatterFn((function(  ) {
  return function(d) { return "Your response helps us improve model performance and the user experience."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1463366044 = messageFormatterFn((function(  ) {
  return function(d) { return "Biased, toxic, or harmful"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a977230117 = messageFormatterFn((function(  ) {
  return function(d) { return "Inaccurate"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a144473386 = messageFormatterFn((function(  ) {
  return function(d) { return "Incomplete"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b766076456 = messageFormatterFn((function(  ) {
  return function(d) { return "Inappropriate style or tone"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a388776964 = messageFormatterFn((function(  ) {
  return function(d) { return "Other"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a314388491 = messageFormatterFn((function(  ) {
  return function(d) { return "Tell us more"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a908941501 = messageFormatterFn((function(  ) {
  return function(d) { return "We value your feedback. Add comments here."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a917117838 = messageFormatterFn((function(  ) {
  return function(d) { return "About"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1873540226 = messageFormatterFn((function(  ) {
  return function(d) { return "Actions"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1650406946 = messageFormatterFn((function(  ) {
  return function(d) { return "All"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a28958017 = messageFormatterFn((function(  ) {
  return function(d) { return "All Labels"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1818813992 = messageFormatterFn((function(  ) {
  return function(d) { return "Search for labels"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1945066211 = messageFormatterFn((function(  ) {
  return function(d) { return "Selected Labels (" + d.total + ")"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a569391674 = messageFormatterFn((function(  ) {
  return function(d) { return "Anytime"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1318586159 = messageFormatterFn((function(  ) {
  return function(d) { return "Apply"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1644559430 = messageFormatterFn((function(  ) {
  return function(d) { return "API Name"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1607297763 = messageFormatterFn((function(  ) {
  return function(d) { return "Author"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b358721508 = messageFormatterFn((function(  ) {
  return function(d) { return "Back"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a903507481 = messageFormatterFn((function(  ) {
  return function(d) { return "Back to calculated insight"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a773572472 = messageFormatterFn((function(  ) {
  return function(d) { return "Back to external assets"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2010595986 = messageFormatterFn((function(  ) {
  return function(d) { return "Back to database"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a434470736 = messageFormatterFn((function(  ) {
  return function(d) { return "Back to fields"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1786019091 = messageFormatterFn((function(  ) {
  return function(d) { return "Back to table"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1380516336 = messageFormatterFn((function(  ) {
  return function(d) { return "Back to data lake"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1128249982 = messageFormatterFn((function(  ) {
  return function(d) { return "Back to data model"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a447950779 = messageFormatterFn((function(  ) {
  return function(d) { return "Calculated Insight"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1001648366 = messageFormatterFn((function(  ) {
  return function(d) { return "Calculated Insights"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a918368879 = messageFormatterFn((function(  ) {
  return function(d) { return "Cancel"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b207712389 = messageFormatterFn((function(  ) {
  return function(d) { return "Certification"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b750601505 = messageFormatterFn((function(  ) {
  return function(d) { return "Certified by"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2088208192 = messageFormatterFn((function(  ) {
  return function(d) { return "Certified filter"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1318931512 = messageFormatterFn((function(  ) {
  return function(d) { return d.characters + " characters remaining"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b604565318 = messageFormatterFn((function(  ) {
  return function(d) { return "Character limit exceeded " + d.total + "/" + d.maximum; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b574174550 = messageFormatterFn((function(  ) {
  return function(d) { return d.total + "/" + d.maximum; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1326383538 = messageFormatterFn((function(  ) {
  return function(d) { return "Clear"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b563676407 = messageFormatterFn((function(  ) {
  return function(d) { return "Clear All"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1624029786 = messageFormatterFn((function(  ) {
  return function(d) { return "Clear filter"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1194760291 = messageFormatterFn((function(  ) {
  return function(d) { return "Clear Filters"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b947702353 = messageFormatterFn((function(  ) {
  return function(d) { return "Abandon changes"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1560472451 = messageFormatterFn((function(  ) {
  return function(d) { return "You've made changes. Are you sure you want to abandon them?"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1939149152 = messageFormatterFn((function(  ) {
  return function(d) { return "Continue working"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a25855358 = messageFormatterFn((function(  ) {
  return function(d) { return "Abandon changes"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1035225929 = messageFormatterFn((function(  ) {
  return function(d) { return "Category selection"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1737651552 = messageFormatterFn((function(  ) {
  return function(d) { return "Label selection"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1870000664 = messageFormatterFn((function(  ) {
  return function(d) { return "Label details"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1377744277 = messageFormatterFn((function( plural, pluralFuncs ) {
  return function(d) { return plural(d.ASSET_COUNT, 0, pluralFuncs.en, { one: function() { return "Data Label: " + d.ASSET_COUNT_LOC + " Selected Asset";}, other: function() { return "Data Labels: " + d.ASSET_COUNT_LOC + " Selected Assets";} }); }
})(messageFormat.plural, {"en": Globalize("en").pluralGenerator()}), Globalize("en").pluralGenerator({}));
Globalize.b1003009865 = messageFormatterFn((function(  ) {
  return function(d) { return "Data Label: (" + d.ASSET_NAME + ")"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1877935322 = messageFormatterFn((function(  ) {
  return function(d) { return "Data Labels"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a437700818 = messageFormatterFn((function(  ) {
  return function(d) { return "Clear Search"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1890493088 = messageFormatterFn((function(  ) {
  return function(d) { return "No labels match your search"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b868384597 = messageFormatterFn((function(  ) {
  return function(d) { return "Select a label to show label details"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1205098945 = messageFormatterFn((function(  ) {
  return function(d) { return d.category + ": " + d.value; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a976901323 = messageFormatterFn((function(  ) {
  return function(d) { return "Added"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b802527118 = messageFormatterFn((function(  ) {
  return function(d) { return "Add Label"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b878801440 = messageFormatterFn((function(  ) {
  return function(d) { return "Delete Label"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a768197211 = messageFormatterFn((function(  ) {
  return function(d) { return "Label category description"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b875639018 = messageFormatterFn((function(  ) {
  return function(d) { return "Label value description"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1515238084 = messageFormatterFn((function(  ) {
  return function(d) { return "Visibility level"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a857775273 = messageFormatterFn((function(  ) {
  return function(d) { return "Message (optional)"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b631222667 = messageFormatterFn((function(  ) {
  return function(d) { return "Certification lets users know that an asset is trusted. The label doesn't show on downstream assets. " + d.learnMoreLink; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1414706985 = messageFormatterFn((function(  ) {
  return function(d) { return "Quality warnings let users know about assets that may have problems. The labels show on the asset and any assets downstream from it. " + d.learnMoreLink; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1333797658 = messageFormatterFn((function(  ) {
  return function(d) { return "Sensitivity labels let users know about assets that need to be treated differently. The labels show on the asset and any assets downstream from it. " + d.learnMoreLink; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1877216276 = messageFormatterFn((function(  ) {
  return function(d) { return "Labels with custom categories are created by an administrator to match your organization's needs. The labels don't show on downstream assets. " + d.learnMoreLink; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1859885090 = messageFormatterFn((function(  ) {
  return function(d) { return d.value + ": " + d.description; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1878607534 = messageFormatterFn((function(  ) {
  return function(d) { return "Failed to save labels"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a196662694 = messageFormatterFn((function( plural, pluralFuncs ) {
  return function(d) { return plural(d.LABEL_COUNT, 0, pluralFuncs.en, { one: function() { return d.LABEL_COUNT_LOC + " label";}, other: function() { return d.LABEL_COUNT_LOC + " labels";} }) + " updated on " + d.CONTENT_TYPE + " " + d.CONTENT_NAME + ". There might be a delay before you see your changes."; }
})(messageFormat.plural, {"en": Globalize("en").pluralGenerator()}), Globalize("en").pluralGenerator({}));
Globalize.a1338838947 = messageFormatterFn((function(  ) {
  return function(d) { return "Close"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1233618208 = messageFormatterFn((function(  ) {
  return function(d) { return ": "; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a418123851 = messageFormatterFn((function(  ) {
  return function(d) { return "Column"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a113328546 = messageFormatterFn((function( plural, pluralFuncs ) {
  return function(d) { return "On " + plural(d.COLUMN_COUNT, 0, pluralFuncs.en, { one: function() { return d.COLUMN_COUNT_LOC + " column in table";}, other: function() { return d.COLUMN_COUNT_LOC + " columns in table";} }); }
})(messageFormat.plural, {"en": Globalize("en").pluralGenerator()}), Globalize("en").pluralGenerator({}));
Globalize.b1886839929 = messageFormatterFn((function(  ) {
  return function(d) { return "Column Description"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1531811917 = messageFormatterFn((function(  ) {
  return function(d) { return "Column Details"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a668036728 = messageFormatterFn((function(  ) {
  return function(d) { return "Certification"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1145008319 = messageFormatterFn((function(  ) {
  return function(d) { return "Data Quality Warning"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b340826147 = messageFormatterFn((function(  ) {
  return function(d) { return "Sensitivity"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a77013598 = messageFormatterFn((function(  ) {
  return function(d) { return "Columns"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b287480959 = messageFormatterFn((function(  ) {
  return function(d) { return "Columns (" + d.count + ")"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b102629065 = messageFormatterFn((function( plural, pluralFuncs ) {
  return function(d) { return "On " + plural(d.COLUMN_COUNT, 0, pluralFuncs.en, { one: function() { return d.COLUMN_COUNT_LOC + " column in table";}, other: function() { return d.COLUMN_COUNT_LOC + " columns";} }); }
})(messageFormat.plural, {"en": Globalize("en").pluralGenerator()}), Globalize("en").pluralGenerator({}));
Globalize.b52883693 = messageFormatterFn((function(  ) {
  return function(d) { return "Combined fields"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1747391126 = messageFormatterFn((function(  ) {
  return function(d) { return "For high visibility warnings, a notification appears when anyone opens a published view affected by this warning."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a401282377 = messageFormatterFn((function(  ) {
  return function(d) { return "High visibility"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1702807324 = messageFormatterFn((function(  ) {
  return function(d) { return "Standard visibility"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a994340807 = messageFormatterFn((function(  ) {
  return function(d) { return "High visibility labels show notifications in affected views and emails."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1532872487 = messageFormatterFn((function(  ) {
  return function(d) { return "Set the visibility level of this label when it's used on assets. The visibility level for a sensitivity label is the same across the entire site, and cannot be set at the asset level."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b260855785 = messageFormatterFn((function(  ) {
  return function(d) { return "Set the default visibility level of this label when it's used on assets. Users with permission can explicitly set the visibility level on individual assets, overriding this default."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1789386642 = messageFormatterFn((function(  ) {
  return function(d) { return "Manage labels"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b256545339 = messageFormatterFn((function(  ) {
  return function(d) { return "Set visibility level"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1393118740 = messageFormatterFn((function(  ) {
  return function(d) { return "Set default visibility level"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a292064395 = messageFormatterFn((function(  ) {
  return function(d) { return "Edit the description of certification labels. "; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a710831998 = messageFormatterFn((function(  ) {
  return function(d) { return "Edit the default visibility level of data quality warnings triggered by flow run monitoring."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a417206751 = messageFormatterFn((function(  ) {
  return function(d) { return "Edit the default visibility level of data quality warnings triggered by extract refresh monitoring."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1232766762 = messageFormatterFn((function(  ) {
  return function(d) { return ", "; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2001984094 = messageFormatterFn((function(  ) {
  return function(d) { return "read more"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b967714201 = messageFormatterFn((function(  ) {
  return function(d) { return "About Tableau AI"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1542723091 = messageFormatterFn((function(  ) {
  return function(d) { return "The description could not be updated. Verify the description is valid and try again."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a783288621 = messageFormatterFn((function(  ) {
  return function(d) { return "The description was successfully updated."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1500083224 = messageFormatterFn((function(  ) {
  return function(d) { return "The description was successfully updated. There might be a delay before you can see your change."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1040558899 = messageFormatterFn((function(  ) {
  return function(d) { return "Tableau couldn't generate a description. Try again later."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a325654510 = messageFormatterFn((function(  ) {
  return function(d) { return "This tool uses generative AI to draft an asset description. Generative AI can produce inaccurate or harmful responses. Check the output before using. " + d.learnMoreLink; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1307616649 = messageFormatterFn((function(  ) {
  return function(d) { return "Connected columns"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1155065247 = messageFormatterFn((function(  ) {
  return function(d) { return "Connected datasource field"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1680884646 = messageFormatterFn((function(  ) {
  return function(d) { return "Connected field"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1480523423 = messageFormatterFn((function(  ) {
  return function(d) { return "Alibaba AnalyticDB for MySQL"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1038687272 = messageFormatterFn((function(  ) {
  return function(d) { return "Aster Database"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1809253508 = messageFormatterFn((function(  ) {
  return function(d) { return "Amazon Athena"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b626103943 = messageFormatterFn((function(  ) {
  return function(d) { return "Amazon Aurora"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a319976559 = messageFormatterFn((function(  ) {
  return function(d) { return "Amazon EMR Hadoop Hive"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1152557331 = messageFormatterFn((function(  ) {
  return function(d) { return "Azure SQL Data Warehouse"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b496408479 = messageFormatterFn((function(  ) {
  return function(d) { return "Google BigQuery"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1254648985 = messageFormatterFn((function(  ) {
  return function(d) { return "IBM BigInsights"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2141939368 = messageFormatterFn((function(  ) {
  return function(d) { return "Box"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b650251252 = messageFormatterFn((function(  ) {
  return function(d) { return "Microsoft Excel (Box)"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1101945754 = messageFormatterFn((function(  ) {
  return function(d) { return "JSON file (Box)"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b267621243 = messageFormatterFn((function(  ) {
  return function(d) { return "Text file (Box)"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1800745157 = messageFormatterFn((function(  ) {
  return function(d) { return "Microsoft Excel (Dropbox)"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1415664361 = messageFormatterFn((function(  ) {
  return function(d) { return "JSON file (Dropbox)"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b87279820 = messageFormatterFn((function(  ) {
  return function(d) { return "Text file (Dropbox)"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1231167962 = messageFormatterFn((function(  ) {
  return function(d) { return "Microsoft Excel (Google Drive)"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1168903796 = messageFormatterFn((function(  ) {
  return function(d) { return "JSON file (Google Drive)"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a400126879 = messageFormatterFn((function(  ) {
  return function(d) { return "Text file (Google Drive)"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b366441349 = messageFormatterFn((function(  ) {
  return function(d) { return "Microsoft Excel (OneDrive)"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a942348247 = messageFormatterFn((function(  ) {
  return function(d) { return "JSON file (OneDrive)"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a356160628 = messageFormatterFn((function(  ) {
  return function(d) { return "Text file (OneDrive)"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1525105948 = messageFormatterFn((function(  ) {
  return function(d) { return "TIBCO Data Virtualization"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2142980131 = messageFormatterFn((function(  ) {
  return function(d) { return "Text File (legacy)"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1567505396 = messageFormatterFn((function(  ) {
  return function(d) { return "Customer Data Platform"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1659693205 = messageFormatterFn((function(  ) {
  return function(d) { return "Databricks"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1008499067 = messageFormatterFn((function(  ) {
  return function(d) { return "Tableau Data Engine"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2143331857 = messageFormatterFn((function(  ) {
  return function(d) { return "IBM DB2"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1122717466 = messageFormatterFn((function(  ) {
  return function(d) { return "Denodo"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1180992249 = messageFormatterFn((function(  ) {
  return function(d) { return "Denormalized Cube"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1442814997 = messageFormatterFn((function(  ) {
  return function(d) { return "Alibaba Data Lake Analytics"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1331307336 = messageFormatterFn((function(  ) {
  return function(d) { return "Apache Drill"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1651561497 = messageFormatterFn((function(  ) {
  return function(d) { return "Dropbox"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a440334067 = messageFormatterFn((function(  ) {
  return function(d) { return "Oracle Essbase"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1782354276 = messageFormatterFn((function(  ) {
  return function(d) { return "Exasol"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b277778412 = messageFormatterFn((function(  ) {
  return function(d) { return "Microsoft Excel (legacy)"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1188266442 = messageFormatterFn((function(  ) {
  return function(d) { return "Microsoft Excel"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a75198500 = messageFormatterFn((function(  ) {
  return function(d) { return "Microsoft Excel Reader"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1402783967 = messageFormatterFn((function(  ) {
  return function(d) { return "Federated"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1891807544 = messageFormatterFn((function(  ) {
  return function(d) { return "Firebird"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1538729361 = messageFormatterFn((function(  ) {
  return function(d) { return "Other Databases (JDBC)"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1395583606 = messageFormatterFn((function(  ) {
  return function(d) { return "Other Databases (ODBC)"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1485995623 = messageFormatterFn((function(  ) {
  return function(d) { return "Google Analytics"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1717326423 = messageFormatterFn((function(  ) {
  return function(d) { return "Google Sheets"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1279121483 = messageFormatterFn((function(  ) {
  return function(d) { return "Google Cloud SQL"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1737610162 = messageFormatterFn((function(  ) {
  return function(d) { return "Google Drive"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2099411084 = messageFormatterFn((function(  ) {
  return function(d) { return "Pivotal Greenplum Database"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a874630894 = messageFormatterFn((function(  ) {
  return function(d) { return "Cloudera Hadoop"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2141847817 = messageFormatterFn((function(  ) {
  return function(d) { return "Apache Hive"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2067718988 = messageFormatterFn((function(  ) {
  return function(d) { return "Hortonworks Hadoop Hive"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1869593975 = messageFormatterFn((function(  ) {
  return function(d) { return "Tableau Data Engine"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1851015651 = messageFormatterFn((function(  ) {
  return function(d) { return "Impala"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1640145674 = messageFormatterFn((function(  ) {
  return function(d) { return "In-memory Federating"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2101076524 = messageFormatterFn((function(  ) {
  return function(d) { return "unknown"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1077626613 = messageFormatterFn((function(  ) {
  return function(d) { return "Kognitio"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a798757065 = messageFormatterFn((function(  ) {
  return function(d) { return "Kyvos"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a162473348 = messageFormatterFn((function(  ) {
  return function(d) { return "MapR Hadoop Hive"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1167203217 = messageFormatterFn((function(  ) {
  return function(d) { return "MariaDB"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1380746894 = messageFormatterFn((function(  ) {
  return function(d) { return "MarkLogic"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b390861756 = messageFormatterFn((function(  ) {
  return function(d) { return "Alibaba MaxCompute"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1653122994 = messageFormatterFn((function(  ) {
  return function(d) { return "MemSQL"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a444726998 = messageFormatterFn((function(  ) {
  return function(d) { return "MonetDB"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a497367695 = messageFormatterFn((function(  ) {
  return function(d) { return "MongoDB BI Connector"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1180142941 = messageFormatterFn((function(  ) {
  return function(d) { return "Microsoft Access"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1243593483 = messageFormatterFn((function(  ) {
  return function(d) { return "Microsoft Analysis Services"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1723920577 = messageFormatterFn((function(  ) {
  return function(d) { return "MySQL"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a429272236 = messageFormatterFn((function(  ) {
  return function(d) { return "unknown"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b574711274 = messageFormatterFn((function(  ) {
  return function(d) { return "IBM PDA (Netezza)"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b566669962 = messageFormatterFn((function(  ) {
  return function(d) { return "OData"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1957930769 = messageFormatterFn((function(  ) {
  return function(d) { return "unknown"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2141266249 = messageFormatterFn((function(  ) {
  return function(d) { return "Spatial File"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b990673056 = messageFormatterFn((function(  ) {
  return function(d) { return "Spatial file"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1423814531 = messageFormatterFn((function(  ) {
  return function(d) { return "OneDrive"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b862059497 = messageFormatterFn((function(  ) {
  return function(d) { return "Oracle"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1514547776 = messageFormatterFn((function(  ) {
  return function(d) { return "Actian Matrix"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2140443633 = messageFormatterFn((function(  ) {
  return function(d) { return "PDF file"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1171778615 = messageFormatterFn((function(  ) {
  return function(d) { return "PDF Reader"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a504780754 = messageFormatterFn((function(  ) {
  return function(d) { return "PostgreSQL"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a654600758 = messageFormatterFn((function(  ) {
  return function(d) { return "Microsoft PowerPivot"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1010291716 = messageFormatterFn((function(  ) {
  return function(d) { return "Presto"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1293281229 = messageFormatterFn((function(  ) {
  return function(d) { return "Progress OpenEdge"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a477592616 = messageFormatterFn((function(  ) {
  return function(d) { return "QuBole Presto"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b233077526 = messageFormatterFn((function(  ) {
  return function(d) { return "Amazon Redshift"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b514848582 = messageFormatterFn((function(  ) {
  return function(d) { return "Tableau Server"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b78537544 = messageFormatterFn((function(  ) {
  return function(d) { return "Salesforce"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1348861733 = messageFormatterFn((function(  ) {
  return function(d) { return "Salesforce Data Cloud"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1384172268 = messageFormatterFn((function(  ) {
  return function(d) { return "SAP NetWeaver Business Warehouse"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1404038763 = messageFormatterFn((function(  ) {
  return function(d) { return "SAP HANA"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1826536758 = messageFormatterFn((function(  ) {
  return function(d) { return ""; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b886207896 = messageFormatterFn((function(  ) {
  return function(d) { return "JSON file"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2014873578 = messageFormatterFn((function(  ) {
  return function(d) { return "SharePoint Lists"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a374126831 = messageFormatterFn((function(  ) {
  return function(d) { return "Snowflake"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b968122694 = messageFormatterFn((function(  ) {
  return function(d) { return "Spark SQL"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a370816572 = messageFormatterFn((function(  ) {
  return function(d) { return "Splunk"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1956764679 = messageFormatterFn((function(  ) {
  return function(d) { return "Tableau Server"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1834174574 = messageFormatterFn((function(  ) {
  return function(d) { return "Microsoft SQL Server"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1828651283 = messageFormatterFn((function(  ) {
  return function(d) { return "Statistical File"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1227308241 = messageFormatterFn((function(  ) {
  return function(d) { return "Statistical file"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1000091655 = messageFormatterFn((function(  ) {
  return function(d) { return "SAP Sybase ASE"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a106520696 = messageFormatterFn((function(  ) {
  return function(d) { return "SAP Sybase IQ"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1816411987 = messageFormatterFn((function(  ) {
  return function(d) { return "Teradata OLAP Connector"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b913266653 = messageFormatterFn((function(  ) {
  return function(d) { return "Teradata"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1960973401 = messageFormatterFn((function(  ) {
  return function(d) { return "Text file"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1067092291 = messageFormatterFn((function(  ) {
  return function(d) { return "Text file"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b202673131 = messageFormatterFn((function(  ) {
  return function(d) { return "Text file reader"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1742481856 = messageFormatterFn((function(  ) {
  return function(d) { return "Actian Vector"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a783567059 = messageFormatterFn((function(  ) {
  return function(d) { return "Vertica"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1077743354 = messageFormatterFn((function(  ) {
  return function(d) { return "VizEngine"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1490974747 = messageFormatterFn((function(  ) {
  return function(d) { return "Web Data Connector"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1185565725 = messageFormatterFn((function(  ) {
  return function(d) { return "Web Data Connector"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1807293023 = messageFormatterFn((function(  ) {
  return function(d) { return "Anaplan"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a920390940 = messageFormatterFn((function(  ) {
  return function(d) { return "Google Ads"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2130622511 = messageFormatterFn((function(  ) {
  return function(d) { return "Intuit QuickBooks Online (9.3-2018.1)"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1889312669 = messageFormatterFn((function(  ) {
  return function(d) { return "Intuit QuickBooks Online"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1103623345 = messageFormatterFn((function(  ) {
  return function(d) { return "LinkedIn Sales Navigator"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a372296395 = messageFormatterFn((function(  ) {
  return function(d) { return "Marketo"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1603923686 = messageFormatterFn((function(  ) {
  return function(d) { return "Oracle Eloqua"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b872595645 = messageFormatterFn((function(  ) {
  return function(d) { return "ServiceNow ITSM"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1294865837 = messageFormatterFn((function(  ) {
  return function(d) { return "Connection Type"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b3403427 = messageFormatterFn((function(  ) {
  return function(d) { return "All"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1501711909 = messageFormatterFn((function(  ) {
  return function(d) { return "Connect"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2018029216 = messageFormatterFn((function(  ) {
  return function(d) { return "Content Type:"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b407079928 = messageFormatterFn((function(  ) {
  return function(d) { return "Search for Data"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2046301251 = messageFormatterFn((function(  ) {
  return function(d) { return "Connecting to this data isn't supported."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1695538266 = messageFormatterFn((function(  ) {
  return function(d) { return "You don't have permissions to connect to this data."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b304711235 = messageFormatterFn((function(  ) {
  return function(d) { return "Add"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1093995766 = messageFormatterFn((function(  ) {
  return function(d) { return "Add label message"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b182573787 = messageFormatterFn((function(  ) {
  return function(d) { return "Delete Label"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a571530649 = messageFormatterFn((function(  ) {
  return function(d) { return "If you remove the label, the message will be deleted and can't be recovered. Are you sure you want to remove the label?"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b278697393 = messageFormatterFn((function(  ) {
  return function(d) { return "Remove Label and Delete Message"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1735634123 = messageFormatterFn((function(  ) {
  return function(d) { return "Edit label message"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1634218805 = messageFormatterFn((function(  ) {
  return function(d) { return "An unexpected error occurred when updating data label for asset \"" + d.CONTENT_NAME + "\"."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1292037973 = messageFormatterFn((function(  ) {
  return function(d) { return "Label message updated. There might be a delay before you see your changes."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1693904309 = messageFormatterFn((function(  ) {
  return function(d) { return "No custom label categories exist. To see more data labels, an administrator must add them first."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a862361764 = messageFormatterFn((function(  ) {
  return function(d) { return "No custom labels info icon"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2079769040 = messageFormatterFn((function(  ) {
  return function(d) { return "Data labels updated on asset \"" + d.CONTENT_NAME + "\". There might be a delay before you see your changes. "; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a550655778 = messageFormatterFn((function(  ) {
  return function(d) { return "More Data Labels"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a65528962 = messageFormatterFn((function(  ) {
  return function(d) { return "Choose other data labels to apply. These labels don't appear on downstream assets."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1655378355 = messageFormatterFn((function(  ) {
  return function(d) { return "Custom Label Dialog Info Tooltip"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a795297500 = messageFormatterFn((function(  ) {
  return function(d) { return "Select…"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2101026795 = messageFormatterFn((function(  ) {
  return function(d) { return "Labels"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a324636312 = messageFormatterFn((function(  ) {
  return function(d) { return "No valid custom labels"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1178158404 = messageFormatterFn((function(  ) {
  return function(d) { return "Applied Labels"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a843351104 = messageFormatterFn((function(  ) {
  return function(d) { return "Messages (optional)"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1456020622 = messageFormatterFn((function(  ) {
  return function(d) { return "Message (optional)"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b686881523 = messageFormatterFn((function(  ) {
  return function(d) { return "Add Message (Optional)"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b907633634 = messageFormatterFn((function(  ) {
  return function(d) { return "Show Custom SQL Query"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1003757240 = messageFormatterFn((function(  ) {
  return function(d) { return "Custom SQL Query"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b316692293 = messageFormatterFn((function(  ) {
  return function(d) { return "Failed to copy to clipboard. Try again."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a333838708 = messageFormatterFn((function(  ) {
  return function(d) { return "Custom SQL is used in this embedded datasource."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b825856525 = messageFormatterFn((function(  ) {
  return function(d) { return "Custom SQL is used in this flow."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1270984580 = messageFormatterFn((function(  ) {
  return function(d) { return "Custom SQL is used in this data source."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1457421050 = messageFormatterFn((function(  ) {
  return function(d) { return "Custom SQL is used in this connection."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1687613406 = messageFormatterFn((function(  ) {
  return function(d) { return "\"" + d.queryName + "\" copied to clipboard."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b870187396 = messageFormatterFn((function(  ) {
  return function(d) { return "Upstream lineage might be incomplete when an embedded data source uses custom SQL."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b821272325 = messageFormatterFn((function(  ) {
  return function(d) { return "Upstream lineage might be incomplete when a flow uses custom SQL."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1694086276 = messageFormatterFn((function(  ) {
  return function(d) { return "Upstream lineage might be incomplete when a data source uses custom SQL."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b582026732 = messageFormatterFn((function(  ) {
  return function(d) { return "Dashboards"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1312675979 = messageFormatterFn((function(  ) {
  return function(d) { return "Dashboards (" + d.count + ")"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1885979312 = messageFormatterFn((function(  ) {
  return function(d) { return "Database"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a940576590 = messageFormatterFn((function(  ) {
  return function(d) { return "Category"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1381919263 = messageFormatterFn((function(  ) {
  return function(d) { return "Grouped database"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a384405531 = messageFormatterFn((function(  ) {
  return function(d) { return "Database Name"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1664107367 = messageFormatterFn((function(  ) {
  return function(d) { return "Databases"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1843544473 = messageFormatterFn((function(  ) {
  return function(d) { return "Databases and Files"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b74303834 = messageFormatterFn((function(  ) {
  return function(d) { return "Databases and Files (" + d.count + ")"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1868620890 = messageFormatterFn((function(  ) {
  return function(d) { return "Databases (" + d.count + ")"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1402023895 = messageFormatterFn((function(  ) {
  return function(d) { return d.databaseName + " (" + d.databaseHostName + ")"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1858253552 = messageFormatterFn((function(  ) {
  return function(d) { return "Database (" + d.type + ")"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b291578800 = messageFormatterFn((function(  ) {
  return function(d) { return "Object API Name"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a415918954 = messageFormatterFn((function(  ) {
  return function(d) { return "CreatedBy"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a312513471 = messageFormatterFn((function(  ) {
  return function(d) { return "Object Label"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1940676918 = messageFormatterFn((function(  ) {
  return function(d) { return "Category"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b265946869 = messageFormatterFn((function(  ) {
  return function(d) { return "Data Cloud Object"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a345657758 = messageFormatterFn((function(  ) {
  return function(d) { return "Data Cloud Objects"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1067754431 = messageFormatterFn((function(  ) {
  return function(d) { return "Data Cloud Objects (" + d.count + ")"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1490796218 = messageFormatterFn((function(  ) {
  return function(d) { return "Information for the Data Details pane is being processed, which may take some time. Try again later."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b58590554 = messageFormatterFn((function(  ) {
  return function(d) { return "Calculation"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a455378887 = messageFormatterFn((function(  ) {
  return function(d) { return "✔ Copied to clipboard"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1358716572 = messageFormatterFn((function(  ) {
  return function(d) { return "Expand Calculation"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a321980334 = messageFormatterFn((function(  ) {
  return function(d) { return "Certified data source (extract)"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1360192331 = messageFormatterFn((function(  ) {
  return function(d) { return "Certified data source (live)"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1189268590 = messageFormatterFn((function(  ) {
  return function(d) { return "Certified virtual connection table (extract)"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1570795531 = messageFormatterFn((function(  ) {
  return function(d) { return "Certified virtual connection table (live)"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1230219751 = messageFormatterFn((function(  ) {
  return function(d) { return "Data source (extract)"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1821155798 = messageFormatterFn((function(  ) {
  return function(d) { return "Data source (live)"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a159038062 = messageFormatterFn((function(  ) {
  return function(d) { return "An unexpected error occurred."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b135046182 = messageFormatterFn((function(  ) {
  return function(d) { return "Data couldn’t be loaded.\n\nIf you are the content owner, try republishing.\n\nIf republishing doesn’t fix the problem contact the Tableau administrator."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1896901555 = messageFormatterFn((function(  ) {
  return function(d) { return "The server encountered an error. Try again."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1362475025 = messageFormatterFn((function(  ) {
  return function(d) { return "The request was for too many results."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b376190499 = messageFormatterFn((function(  ) {
  return function(d) { return "Request size limit exceeded"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a74688109 = messageFormatterFn((function(  ) {
  return function(d) { return "Data Details"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b194579532 = messageFormatterFn((function(  ) {
  return function(d) { return "Project"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1469831884 = messageFormatterFn((function(  ) {
  return function(d) { return "Try again later."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1525977760 = messageFormatterFn((function(  ) {
  return function(d) { return "The server is busy."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2093904506 = messageFormatterFn((function(  ) {
  return function(d) { return "Can’t show all results from the query.\n\nTry again later."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b602607438 = messageFormatterFn((function(  ) {
  return function(d) { return "Timeout limit exceeded"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a647811673 = messageFormatterFn((function(  ) {
  return function(d) { return "Virtual connection table (extract)"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a879643114 = messageFormatterFn((function(  ) {
  return function(d) { return "Virtual connection table (live)"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1325202285 = messageFormatterFn((function(  ) {
  return function(d) { return "Data Lake Object"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1868326020 = messageFormatterFn((function(  ) {
  return function(d) { return "Data Lake Objects"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1343349247 = messageFormatterFn((function(  ) {
  return function(d) { return "Data Model Object"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1305770198 = messageFormatterFn((function(  ) {
  return function(d) { return "Data Model Objects"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1509527073 = messageFormatterFn((function(  ) {
  return function(d) { return "Set by " + d.name; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1767697189 = messageFormatterFn((function(  ) {
  return function(d) { return "Certified by " + d.name; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1076552907 = messageFormatterFn((function(  ) {
  return function(d) { return "On this asset"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b77722308 = messageFormatterFn((function(  ) {
  return function(d) { return "Upstream from this asset"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a761228366 = messageFormatterFn((function( plural, pluralFuncs ) {
  return function(d) { return plural(d.MESSAGE_COUNT, 0, pluralFuncs.en, { one: function() { return d.MESSAGE_COUNT_LOC + " data quality warning";}, other: function() { return d.MESSAGE_COUNT_LOC + " data quality warnings";} }); }
})(messageFormat.plural, {"en": Globalize("en").pluralGenerator()}), Globalize("en").pluralGenerator({}));
Globalize.b1273282608 = messageFormatterFn((function(  ) {
  return function(d) { return "On this column"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1805168168 = messageFormatterFn((function(  ) {
  return function(d) { return "On this connection"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b57989899 = messageFormatterFn((function(  ) {
  return function(d) { return "On this database"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b719354497 = messageFormatterFn((function(  ) {
  return function(d) { return "On this data source"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1008620696 = messageFormatterFn((function(  ) {
  return function(d) { return "On this flow"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1989512342 = messageFormatterFn((function(  ) {
  return function(d) { return "On this table"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a651507907 = messageFormatterFn((function(  ) {
  return function(d) { return "Upstream from this connection"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1936995680 = messageFormatterFn((function(  ) {
  return function(d) { return "Upstream from this database"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1737321578 = messageFormatterFn((function(  ) {
  return function(d) { return "Upstream from this data source"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2016833483 = messageFormatterFn((function(  ) {
  return function(d) { return "Upstream from this field"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a483778195 = messageFormatterFn((function(  ) {
  return function(d) { return "Upstream from this flow"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1407760363 = messageFormatterFn((function(  ) {
  return function(d) { return "Upstream from this metric"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1325180319 = messageFormatterFn((function(  ) {
  return function(d) { return "Upstream from this table"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a938776138 = messageFormatterFn((function(  ) {
  return function(d) { return "Upstream from this view"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a196921055 = messageFormatterFn((function(  ) {
  return function(d) { return "Upstream from this workbook"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b361209651 = messageFormatterFn((function(  ) {
  return function(d) { return "Certified"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a270547363 = messageFormatterFn((function(  ) {
  return function(d) { return "Deprecated"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1545099830 = messageFormatterFn((function(  ) {
  return function(d) { return "Extract refresh failed"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1119462595 = messageFormatterFn((function(  ) {
  return function(d) { return "Flow run failed"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1675498809 = messageFormatterFn((function(  ) {
  return function(d) { return "Under maintenance"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b807679098 = messageFormatterFn((function(  ) {
  return function(d) { return "Sensitive data"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a187156607 = messageFormatterFn((function(  ) {
  return function(d) { return "Stale data"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a46252290 = messageFormatterFn((function(  ) {
  return function(d) { return "Warning"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a631077276 = messageFormatterFn((function(  ) {
  return function(d) { return "By " + d.displayName + " on " + d.updatedAt; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1740802482 = messageFormatterFn((function(  ) {
  return function(d) { return "Quality Warning…"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a880355276 = messageFormatterFn((function(  ) {
  return function(d) { return "All assets"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1524434255 = messageFormatterFn((function(  ) {
  return function(d) { return "Assets without warnings"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1489196049 = messageFormatterFn((function(  ) {
  return function(d) { return "Assets with warnings"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b801460515 = messageFormatterFn((function(  ) {
  return function(d) { return "Warnings"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b920575950 = messageFormatterFn((function(  ) {
  return function(d) { return "Quality Warning"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b875716486 = messageFormatterFn((function(  ) {
  return function(d) { return "Data Source"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a528752562 = messageFormatterFn((function(  ) {
  return function(d) { return "Connects to"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a809105089 = messageFormatterFn((function(  ) {
  return function(d) { return "Last Refreshed"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1268987786 = messageFormatterFn((function(  ) {
  return function(d) { return "Extract"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a568679905 = messageFormatterFn((function(  ) {
  return function(d) { return "Live Connection"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1422835186 = messageFormatterFn((function(  ) {
  return function(d) { return "Data Source Place Page"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1377331185 = messageFormatterFn((function(  ) {
  return function(d) { return "Data Sources"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a923256048 = messageFormatterFn((function(  ) {
  return function(d) { return "Data Sources (" + d.count + ")"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2009422442 = messageFormatterFn((function(  ) {
  return function(d) { return "Data Streams"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a604453815 = messageFormatterFn((function(  ) {
  return function(d) { return "Data Source with high severity warning"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1333481326 = messageFormatterFn((function(  ) {
  return function(d) { return "Data Source with warning"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1179791014 = messageFormatterFn((function(  ) {
  return function(d) { return "Modified"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a793501184 = messageFormatterFn((function(  ) {
  return function(d) { return "There are \"" + d.NAME + "\" labels attached to " + d.ITEM_COUNT + " assets. If you delete the \"" + d.NAME + "\" label, you'll also remove the label from those assets. Are you sure you want to delete the label?"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b8626824 = messageFormatterFn((function(  ) {
  return function(d) { return "Are you sure you want to delete the label?"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a403992509 = messageFormatterFn((function(  ) {
  return function(d) { return "Description"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b910233470 = messageFormatterFn((function(  ) {
  return function(d) { return "Description (Data Source)"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1482218507 = messageFormatterFn((function(  ) {
  return function(d) { return "Description is inherited"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1940626843 = messageFormatterFn((function(  ) {
  return function(d) { return "Description inherited from column " + d.column + " in table " + d.table + "."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1802108016 = messageFormatterFn((function(  ) {
  return function(d) { return "Description inherited from column " + d.column + " in table " + d.table + ", through field " + d.field + " in data source " + d.datasource + "."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b903744415 = messageFormatterFn((function(  ) {
  return function(d) { return "Description inherited from field " + d.field + " in data source " + d.datasource + "."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1853666447 = messageFormatterFn((function(  ) {
  return function(d) { return "Inherited from column"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1997453218 = messageFormatterFn((function(  ) {
  return function(d) { return "in data source"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a761850505 = messageFormatterFn((function(  ) {
  return function(d) { return "in table"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b221543209 = messageFormatterFn((function(  ) {
  return function(d) { return "Description (Workbook)"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1520289324 = messageFormatterFn((function(  ) {
  return function(d) { return "Set a warning label to show on this asset and any asset downstream from it. An administrator can customize label values and descriptions to match your organization's needs. "; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1523835615 = messageFormatterFn((function(  ) {
  return function(d) { return "Certify this asset to let users know it's trusted. An administrator can customize the certification label description to match your organization's needs. "; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1930775044 = messageFormatterFn((function(  ) {
  return function(d) { return "Set a sensitivity label to show on this asset and any asset downstream from it. An administrator can customize label values and descriptions to match your organization's needs. "; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b925561586 = messageFormatterFn((function(  ) {
  return function(d) { return "Generate a quality warning when an extract refresh fails. "; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a40899845 = messageFormatterFn((function(  ) {
  return function(d) { return "Generate a quality warning when one or more flow outputs fail. "; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1461221245 = messageFormatterFn((function(  ) {
  return function(d) { return "Set a warning label to show on this asset and any asset downstream from it. "; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1476466258 = messageFormatterFn((function(  ) {
  return function(d) { return "Dialog help message popover"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1293470152 = messageFormatterFn((function(  ) {
  return function(d) { return "Dialog info icon"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a339978002 = messageFormatterFn((function(  ) {
  return function(d) { return "Copy"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1655493588 = messageFormatterFn((function(  ) {
  return function(d) { return "Learn more"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1230356124 = messageFormatterFn((function(  ) {
  return function(d) { return "data source"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1694911364 = messageFormatterFn((function(  ) {
  return function(d) { return "view"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2133358932 = messageFormatterFn((function(  ) {
  return function(d) { return "virtual connection"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1700812882 = messageFormatterFn((function(  ) {
  return function(d) { return "database"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b207720911 = messageFormatterFn((function(  ) {
  return function(d) { return "table"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1768677893 = messageFormatterFn((function(  ) {
  return function(d) { return "flow"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b791585571 = messageFormatterFn((function(  ) {
  return function(d) { return "virtual connection table"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1964522925 = messageFormatterFn((function(  ) {
  return function(d) { return "column"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b780645952 = messageFormatterFn((function(  ) {
  return function(d) { return "columns"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1349330093 = messageFormatterFn((function(  ) {
  return function(d) { return "asset"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b430874406 = messageFormatterFn((function(  ) {
  return function(d) { return "assets"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2064980303 = messageFormatterFn((function(  ) {
  return function(d) { return "data sources"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1476975084 = messageFormatterFn((function(  ) {
  return function(d) { return "This asset is certified"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1741008081 = messageFormatterFn((function(  ) {
  return function(d) { return "views"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a599271868 = messageFormatterFn((function(  ) {
  return function(d) { return "tables"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1738092648 = messageFormatterFn((function(  ) {
  return function(d) { return "flows"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b365722693 = messageFormatterFn((function(  ) {
  return function(d) { return "databases"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1034035361 = messageFormatterFn((function(  ) {
  return function(d) { return "virtual connections"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b729802328 = messageFormatterFn((function(  ) {
  return function(d) { return "virtual connection tables"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a343388971 = messageFormatterFn((function(  ) {
  return function(d) { return "Certification"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a205321190 = messageFormatterFn((function(  ) {
  return function(d) { return "Sensitivity Label"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a202460911 = messageFormatterFn((function(  ) {
  return function(d) { return "Extract Refresh Monitoring"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b285921760 = messageFormatterFn((function(  ) {
  return function(d) { return "Flow Run Monitoring"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a414678698 = messageFormatterFn((function(  ) {
  return function(d) { return "Data Quality Warning"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1469430413 = messageFormatterFn((function(  ) {
  return function(d) { return "Show label"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1445804166 = messageFormatterFn((function(  ) {
  return function(d) { return "Extract refresh monitoring"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1757463971 = messageFormatterFn((function(  ) {
  return function(d) { return "Flow run monitoring"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a43394995 = messageFormatterFn((function(  ) {
  return function(d) { return "Show warning"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1376195242 = messageFormatterFn((function(  ) {
  return function(d) { return "Downstream Dashboards"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b586911863 = messageFormatterFn((function(  ) {
  return function(d) { return "Downstream Dashboards (" + d.count + ")"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b581347049 = messageFormatterFn((function(  ) {
  return function(d) { return "Downstream Databases"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b682971928 = messageFormatterFn((function(  ) {
  return function(d) { return "Downstream Databases (" + d.count + ")"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b226751219 = messageFormatterFn((function(  ) {
  return function(d) { return "Downstream Published Data Sources"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2129392206 = messageFormatterFn((function(  ) {
  return function(d) { return "Downstream Published Data Sources (" + d.count + ")"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b426628476 = messageFormatterFn((function(  ) {
  return function(d) { return "Downstream Flows"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1515051493 = messageFormatterFn((function(  ) {
  return function(d) { return "Downstream Flows (" + d.count + ")"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b423876125 = messageFormatterFn((function(  ) {
  return function(d) { return "Downstream Lenses"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1835531876 = messageFormatterFn((function(  ) {
  return function(d) { return "Downstream Lenses (" + d.count + ")"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1057993278 = messageFormatterFn((function(  ) {
  return function(d) { return "Downstream Metrics"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b917190499 = messageFormatterFn((function(  ) {
  return function(d) { return "Downstream Metrics (" + d.count + ")"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1715506662 = messageFormatterFn((function(  ) {
  return function(d) { return "Downstream Metric Definitions"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1232271547 = messageFormatterFn((function(  ) {
  return function(d) { return "Downstream Metric Definitions (" + d.count + ")"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b707757353 = messageFormatterFn((function(  ) {
  return function(d) { return "Downstream Owners"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a593330984 = messageFormatterFn((function(  ) {
  return function(d) { return "Downstream Owners (" + d.count + ")"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1283260587 = messageFormatterFn((function(  ) {
  return function(d) { return "Downstream Sheets"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b490189868 = messageFormatterFn((function(  ) {
  return function(d) { return "Downstream Sheets (" + d.count + ")"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1028642780 = messageFormatterFn((function(  ) {
  return function(d) { return "Downstream Tables"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a776338371 = messageFormatterFn((function(  ) {
  return function(d) { return "Downstream Tables (" + d.count + ")"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1975206215 = messageFormatterFn((function(  ) {
  return function(d) { return "Downstream Tables and Objects"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a316308870 = messageFormatterFn((function(  ) {
  return function(d) { return "Downstream Tables and Objects (" + d.count + ")"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a192044743 = messageFormatterFn((function(  ) {
  return function(d) { return "Users"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1243519873 = messageFormatterFn((function(  ) {
  return function(d) { return "Downstream Virtual Connections"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a230195646 = messageFormatterFn((function(  ) {
  return function(d) { return "Downstream Virtual Connections (" + d.count + ")"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a855201357 = messageFormatterFn((function(  ) {
  return function(d) { return "Downstream Virtual Connection Tables"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2133513102 = messageFormatterFn((function(  ) {
  return function(d) { return "Downstream Virtual Connection Tables (" + d.count + ")"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1145562296 = messageFormatterFn((function(  ) {
  return function(d) { return "Downstream Workbooks"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b558324121 = messageFormatterFn((function(  ) {
  return function(d) { return "Downstream Workbooks (" + d.count + ")"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b92584545 = messageFormatterFn((function(  ) {
  return function(d) { return "Data Quality Warnings"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1243627219 = messageFormatterFn((function(  ) {
  return function(d) { return "Edit description"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1265436184 = messageFormatterFn((function(  ) {
  return function(d) { return "View description"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1370129977 = messageFormatterFn((function(  ) {
  return function(d) { return "Edit"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b772221886 = messageFormatterFn((function(  ) {
  return function(d) { return "Draft with Einstein"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a994316214 = messageFormatterFn((function(  ) {
  return function(d) { return "Tableau AI is temporarily unavailable. Try again later."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b922719679 = messageFormatterFn((function(  ) {
  return function(d) { return "Tableau Catalog can't find the asset. Try again later."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1789062517 = messageFormatterFn((function(  ) {
  return function(d) { return "Describe the asset, or have Einstein draft a description for you."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1722147578 = messageFormatterFn((function(  ) {
  return function(d) { return "Describe the asset."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1991568744 = messageFormatterFn((function(  ) {
  return function(d) { return "Asset Description"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a735332777 = messageFormatterFn((function(  ) {
  return function(d) { return "Edit Description…"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1627823005 = messageFormatterFn((function(  ) {
  return function(d) { return "Edit Label"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2147426184 = messageFormatterFn((function(  ) {
  return function(d) { return "Edit tags"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2019832233 = messageFormatterFn((function(  ) {
  return function(d) { return "New Message"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b763123478 = messageFormatterFn((function(  ) {
  return function(d) { return "Message"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b581554139 = messageFormatterFn((function(  ) {
  return function(d) { return "Subject"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a41672263 = messageFormatterFn((function(  ) {
  return function(d) { return "Embedded Asset"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1083859701 = messageFormatterFn((function(  ) {
  return function(d) { return "This table or file is embedded in the published asset on the server, and you can't create a new workbook from it. Files embedded in workbooks aren't shared with other Tableau site users."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1166477403 = messageFormatterFn((function(  ) {
  return function(d) { return "Embedded Data Sources"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a563711514 = messageFormatterFn((function(  ) {
  return function(d) { return "Embedded Data Sources (" + d.count + ")"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1822263174 = messageFormatterFn((function(  ) {
  return function(d) { return "This column is not used by any fields."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a888651028 = messageFormatterFn((function(  ) {
  return function(d) { return "This column is not used by any flows."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b434598895 = messageFormatterFn((function(  ) {
  return function(d) { return "You've filtered out all the content."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1808287581 = messageFormatterFn((function(  ) {
  return function(d) { return "There is nothing here yet."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1610342908 = messageFormatterFn((function(  ) {
  return function(d) { return "There is nothing here yet."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b482608059 = messageFormatterFn((function(  ) {
  return function(d) { return "There are no downstream dashboards."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a907021620 = messageFormatterFn((function(  ) {
  return function(d) { return "There are no downstream databases."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b433094166 = messageFormatterFn((function(  ) {
  return function(d) { return "There are no downstream data sources."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b778298564 = messageFormatterFn((function(  ) {
  return function(d) { return "With the current filter applied, there are no downstream dashboards."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1313125213 = messageFormatterFn((function(  ) {
  return function(d) { return "With the current filter applied, there are no downstream databases."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1009565229 = messageFormatterFn((function(  ) {
  return function(d) { return "With the current filter applied, there are no downstream data sources."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1380676726 = messageFormatterFn((function(  ) {
  return function(d) { return "With the current filter applied, there are no downstream flows."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1211598473 = messageFormatterFn((function(  ) {
  return function(d) { return "With the current filter applied, there are no downstream lenses."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1227316917 = messageFormatterFn((function(  ) {
  return function(d) { return "With the current filter applied, there are no downstream metric definitions."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a658655048 = messageFormatterFn((function(  ) {
  return function(d) { return "With the current filter applied, there are no downstream metrics."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1597833853 = messageFormatterFn((function(  ) {
  return function(d) { return "With the current filter applied, there are no downstream owners."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b657769647 = messageFormatterFn((function(  ) {
  return function(d) { return "With the current filter applied, there are no downstream sheets."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1859323970 = messageFormatterFn((function(  ) {
  return function(d) { return "With the current filter applied, there are no downstream tables."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1864762535 = messageFormatterFn((function(  ) {
  return function(d) { return "With the current filter applied, there are no downstream virtual connections."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b876231917 = messageFormatterFn((function(  ) {
  return function(d) { return "With the current filter applied, there are no downstream virtual connection tables."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b252839746 = messageFormatterFn((function(  ) {
  return function(d) { return "With the current filter applied, there are no downstream workbooks."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1508665569 = messageFormatterFn((function(  ) {
  return function(d) { return "There are no downstream flows."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a586896402 = messageFormatterFn((function(  ) {
  return function(d) { return "There are no downstream lenses."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2046627444 = messageFormatterFn((function(  ) {
  return function(d) { return "There are no downstream metric definitions."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1527239969 = messageFormatterFn((function(  ) {
  return function(d) { return "There are no downstream metrics."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a973131782 = messageFormatterFn((function(  ) {
  return function(d) { return "There are no downstream owners."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1282471718 = messageFormatterFn((function(  ) {
  return function(d) { return "There are no downstream sheets."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1234621899 = messageFormatterFn((function(  ) {
  return function(d) { return "There are no downstream tables."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b843264848 = messageFormatterFn((function(  ) {
  return function(d) { return "There are no downstream virtual connections."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1938266282 = messageFormatterFn((function(  ) {
  return function(d) { return "There are no downstream virtual connection tables."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b658943339 = messageFormatterFn((function(  ) {
  return function(d) { return "There are no downstream workbooks."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b993416866 = messageFormatterFn((function(  ) {
  return function(d) { return "This field does not reference any fields."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b271961894 = messageFormatterFn((function(  ) {
  return function(d) { return "This field is not used by any columns."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a245855681 = messageFormatterFn((function(  ) {
  return function(d) { return "Lineage can't be determined until the flow is successfully run."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a194195419 = messageFormatterFn((function(  ) {
  return function(d) { return "There are no upstream databases."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1735694033 = messageFormatterFn((function(  ) {
  return function(d) { return "There are no upstream data sources."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1070973180 = messageFormatterFn((function(  ) {
  return function(d) { return "With the current filter applied, there are no upstream databases."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1384415162 = messageFormatterFn((function(  ) {
  return function(d) { return "With the current filter applied, there are no upstream data sources."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b234356047 = messageFormatterFn((function(  ) {
  return function(d) { return "With the current filter applied, there are no upstream flows."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1907166142 = messageFormatterFn((function(  ) {
  return function(d) { return "With the current filter applied, there are no upstream lenses."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1259440645 = messageFormatterFn((function(  ) {
  return function(d) { return "With the current filter applied, there are no upstream tables."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a89170239 = messageFormatterFn((function(  ) {
  return function(d) { return "With the current filter applied, there are no upstream data cloud Objects."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1959418656 = messageFormatterFn((function(  ) {
  return function(d) { return "With the current filter applied, there are no upstream virtual connections."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1237212794 = messageFormatterFn((function(  ) {
  return function(d) { return "With the current filter applied, there are no upstream virtual connection tables."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1505691896 = messageFormatterFn((function(  ) {
  return function(d) { return "There are no upstream flows."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1631095499 = messageFormatterFn((function(  ) {
  return function(d) { return "There are no upstream lenses."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2016146300 = messageFormatterFn((function(  ) {
  return function(d) { return "There are no upstream tables."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b317300600 = messageFormatterFn((function(  ) {
  return function(d) { return "There are no upstream data cloud objects."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1735870999 = messageFormatterFn((function(  ) {
  return function(d) { return "There are no upstream virtual connections."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1442215151 = messageFormatterFn((function(  ) {
  return function(d) { return "There are no upstream virtual connection tables."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b60559987 = messageFormatterFn((function(  ) {
  return function(d) { return "Lineage isn't available for virtual connections with no tables."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1360032964 = messageFormatterFn((function(  ) {
  return function(d) { return "Lineage isn't available for workbooks with no embedded datasources."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1981589857 = messageFormatterFn((function(  ) {
  return function(d) { return "enter search"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a632821577 = messageFormatterFn((function(  ) {
  return function(d) { return "Error"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1742419458 = messageFormatterFn((function(  ) {
  return function(d) { return "Still loading lineage information. Try again later."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b346296421 = messageFormatterFn((function(  ) {
  return function(d) { return "Showing partial results. Still loading lineage information. Try again later."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a762290911 = messageFormatterFn((function(  ) {
  return function(d) { return "Tableau Catalog isn't fully configured. Ask your Tableau administrator to run the “tsm maintenance metadata-services enable” command."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a699200526 = messageFormatterFn((function(  ) {
  return function(d) { return "Information for this page not available. If you are the content owner, try republishing."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1306509739 = messageFormatterFn((function(  ) {
  return function(d) { return "The server encountered an error. Try again."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1690788491 = messageFormatterFn((function(  ) {
  return function(d) { return "Information for this page not found. It may still be loading, or you don't have permissions to view it."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b852739276 = messageFormatterFn((function(  ) {
  return function(d) { return "Information for this page could not be loaded."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1775917465 = messageFormatterFn((function(  ) {
  return function(d) { return "Request size limit exceeded."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a201749092 = messageFormatterFn((function(  ) {
  return function(d) { return "Showing partial results. Request size limit exceeded."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2111379964 = messageFormatterFn((function(  ) {
  return function(d) { return "Request rate limit exceeded. Try again later."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a305100769 = messageFormatterFn((function(  ) {
  return function(d) { return "Showing partial results. Request rate limit exceeded. Try again later."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b634258924 = messageFormatterFn((function(  ) {
  return function(d) { return "Your session timed out. Reopen the dialog to connect."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1986405742 = messageFormatterFn((function(  ) {
  return function(d) { return "Request time limit exceeded. Try again later."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b241581137 = messageFormatterFn((function(  ) {
  return function(d) { return "Showing partial results. Request time limit exceeded. Try again later."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1700376490 = messageFormatterFn((function(  ) {
  return function(d) { return "An unexpected error occurred."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a413830179 = messageFormatterFn((function(  ) {
  return function(d) { return "External Assets"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1208902848 = messageFormatterFn((function(  ) {
  return function(d) { return "Extract"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1682007621 = messageFormatterFn((function(  ) {
  return function(d) { return "Boolean Field"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a900044410 = messageFormatterFn((function(  ) {
  return function(d) { return "Boolean Calculated Field"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a485204840 = messageFormatterFn((function(  ) {
  return function(d) { return "Boolean"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1589699076 = messageFormatterFn((function(  ) {
  return function(d) { return "Date"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b58603247 = messageFormatterFn((function(  ) {
  return function(d) { return "Datetime"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1080767746 = messageFormatterFn((function(  ) {
  return function(d) { return "Integer"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1993641972 = messageFormatterFn((function(  ) {
  return function(d) { return "Real"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2036331196 = messageFormatterFn((function(  ) {
  return function(d) { return "Spatial"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1303782681 = messageFormatterFn((function(  ) {
  return function(d) { return "String"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b960131378 = messageFormatterFn((function(  ) {
  return function(d) { return "Table"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b374619064 = messageFormatterFn((function(  ) {
  return function(d) { return "Tuple"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b427400438 = messageFormatterFn((function(  ) {
  return function(d) { return "Unknown"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a336900077 = messageFormatterFn((function(  ) {
  return function(d) { return "Continuous Date Calculated Field"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b512504297 = messageFormatterFn((function(  ) {
  return function(d) { return "Discrete Date Calculated Field"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2054127624 = messageFormatterFn((function(  ) {
  return function(d) { return "Continuous Date Field"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b416668446 = messageFormatterFn((function(  ) {
  return function(d) { return "Discrete Date Field"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1721907750 = messageFormatterFn((function(  ) {
  return function(d) { return "Continuous Date & Time Calculated Field"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a133397316 = messageFormatterFn((function(  ) {
  return function(d) { return "Discrete Date & Time Calculated Field"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1669098395 = messageFormatterFn((function(  ) {
  return function(d) { return "Continuous Date & Time Field"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2034143089 = messageFormatterFn((function(  ) {
  return function(d) { return "Discrete Date & Time Field"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b368822051 = messageFormatterFn((function(  ) {
  return function(d) { return "Field details"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a990260768 = messageFormatterFn((function(  ) {
  return function(d) { return "Aggregation"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b51913949 = messageFormatterFn((function(  ) {
  return function(d) { return "Aggregation param"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1762423742 = messageFormatterFn((function(  ) {
  return function(d) { return "Autogenerated?"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a429128934 = messageFormatterFn((function(  ) {
  return function(d) { return "Bin size"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1004169561 = messageFormatterFn((function(  ) {
  return function(d) { return "Combination type"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1653796448 = messageFormatterFn((function(  ) {
  return function(d) { return "Data category"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1889833700 = messageFormatterFn((function(  ) {
  return function(d) { return "Data type"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a396767894 = messageFormatterFn((function(  ) {
  return function(d) { return "Default format"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b645550907 = messageFormatterFn((function(  ) {
  return function(d) { return "Delimiter"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a630443078 = messageFormatterFn((function(  ) {
  return function(d) { return "Folder"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b628727516 = messageFormatterFn((function(  ) {
  return function(d) { return "Formula"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1981189934 = messageFormatterFn((function(  ) {
  return function(d) { return "Has 'Other'?"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1982060766 = messageFormatterFn((function(  ) {
  return function(d) { return "Has user reference?"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1437995390 = messageFormatterFn((function(  ) {
  return function(d) { return "Hidden?"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b995382439 = messageFormatterFn((function(  ) {
  return function(d) { return "No"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1331913102 = messageFormatterFn((function(  ) {
  return function(d) { return "Role"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a668623348 = messageFormatterFn((function(  ) {
  return function(d) { return "Semantic role"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b782147611 = messageFormatterFn((function(  ) {
  return function(d) { return "Yes"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1585828662 = messageFormatterFn((function(  ) {
  return function(d) { return "Filter by field name"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a904223440 = messageFormatterFn((function(  ) {
  return function(d) { return "Continuous Geographic Calculated Field"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b534260294 = messageFormatterFn((function(  ) {
  return function(d) { return "Discrete Geographic Calculated Field"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b136833957 = messageFormatterFn((function(  ) {
  return function(d) { return "Continuous Geographic Field"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1308527099 = messageFormatterFn((function(  ) {
  return function(d) { return "Discrete Geographic Field"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a831408602 = messageFormatterFn((function(  ) {
  return function(d) { return "Group Field"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b548217427 = messageFormatterFn((function(  ) {
  return function(d) { return "Geographic Group Field"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a919437328 = messageFormatterFn((function(  ) {
  return function(d) { return "Hierarchy Field"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b23464647 = messageFormatterFn((function(  ) {
  return function(d) { return d.FieldName + " (Count)"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2039688584 = messageFormatterFn((function(  ) {
  return function(d) { return "Continuous Number Calculated Field"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1411053170 = messageFormatterFn((function(  ) {
  return function(d) { return "Discrete Number Calculated Field"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b385088749 = messageFormatterFn((function(  ) {
  return function(d) { return "Continuous Number Field"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1985065661 = messageFormatterFn((function(  ) {
  return function(d) { return "Discrete Number Field"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b223098620 = messageFormatterFn((function(  ) {
  return function(d) { return "Continuous Numeric Bin Field"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a979648750 = messageFormatterFn((function(  ) {
  return function(d) { return "Discrete Numeric Bin Field"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a353122136 = messageFormatterFn((function(  ) {
  return function(d) { return "Dimension"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2052873456 = messageFormatterFn((function(  ) {
  return function(d) { return "Measure"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b690643204 = messageFormatterFn((function(  ) {
  return function(d) { return "Unknown"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1177027412 = messageFormatterFn((function(  ) {
  return function(d) { return "Nominal"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1448616703 = messageFormatterFn((function(  ) {
  return function(d) { return "Ordinal"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1797085091 = messageFormatterFn((function(  ) {
  return function(d) { return "Quantitative"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b22430566 = messageFormatterFn((function(  ) {
  return function(d) { return "Unknown"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a112553710 = messageFormatterFn((function(  ) {
  return function(d) { return "Fields"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a762399485 = messageFormatterFn((function(  ) {
  return function(d) { return "Set Field"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1520158479 = messageFormatterFn((function(  ) {
  return function(d) { return "Fields (" + d.count + ")"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1860753556 = messageFormatterFn((function(  ) {
  return function(d) { return "Continuous Text Calculated Field"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a928782678 = messageFormatterFn((function(  ) {
  return function(d) { return "Discrete Text Calculated Field"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2025812233 = messageFormatterFn((function(  ) {
  return function(d) { return "Continuous Text Field"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1757419615 = messageFormatterFn((function(  ) {
  return function(d) { return "Discrete Text Field"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1544908571 = messageFormatterFn((function(  ) {
  return function(d) { return "Unknown"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1526167450 = messageFormatterFn((function(  ) {
  return function(d) { return "This file type can contain multiple tables."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1837890652 = messageFormatterFn((function(  ) {
  return function(d) { return "File Name"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1895360374 = messageFormatterFn((function(  ) {
  return function(d) { return "File Path"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1256824824 = messageFormatterFn((function(  ) {
  return function(d) { return "Files"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b767877493 = messageFormatterFn((function(  ) {
  return function(d) { return "This file type can only contain a single table."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2031919435 = messageFormatterFn((function(  ) {
  return function(d) { return "File Type"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1713170767 = messageFormatterFn((function(  ) {
  return function(d) { return "File (" + d.type + ")"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a320374765 = messageFormatterFn((function(  ) {
  return function(d) { return "Filter: " + d.filteredItemName; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1355492276 = messageFormatterFn((function(  ) {
  return function(d) { return "Filtered"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b298590559 = messageFormatterFn((function(  ) {
  return function(d) { return "You've filtered out all the content"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1341759228 = messageFormatterFn((function(  ) {
  return function(d) { return "Filters: " + d.count; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b233677149 = messageFormatterFn((function(  ) {
  return function(d) { return "Flow"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2092767973 = messageFormatterFn((function(  ) {
  return function(d) { return "Output column information is not available."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1745807905 = messageFormatterFn((function(  ) {
  return function(d) { return "Output field information is not available."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a668175535 = messageFormatterFn((function(  ) {
  return function(d) { return "Outputs to column"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1951357143 = messageFormatterFn((function(  ) {
  return function(d) { return "Outputs to field"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1491264798 = messageFormatterFn((function(  ) {
  return function(d) { return "Flow parameters are not yet supported in the lineage graph. Only the last flow run is reflected."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2001773950 = messageFormatterFn((function(  ) {
  return function(d) { return "Flow Using This Database"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b626967489 = messageFormatterFn((function(  ) {
  return function(d) { return "Flow Using This Table"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1346019078 = messageFormatterFn((function(  ) {
  return function(d) { return "Flows"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b308829223 = messageFormatterFn((function(  ) {
  return function(d) { return "Flows (" + d.count + ")"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b754135249 = messageFormatterFn((function(  ) {
  return function(d) { return "Full Name"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b259263846 = messageFormatterFn((function(  ) {
  return function(d) { return "You’re about to use Einstein"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1685695163 = messageFormatterFn((function(  ) {
  return function(d) { return "Generative AI can produce inaccurate or harmful responses. Review output for accuracy and safety. You assume responsibility for how the outcomes of Einstein are applied to your organization. " + d.learnMoreLink; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1585917174 = messageFormatterFn((function(  ) {
  return function(d) { return "content item"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1972308904 = messageFormatterFn((function(  ) {
  return function(d) { return "Got It"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b156280644 = messageFormatterFn((function(  ) {
  return function(d) { return " " + d.count + "+"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1404764277 = messageFormatterFn((function(  ) {
  return function(d) { return "Certified"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b944437589 = messageFormatterFn((function(  ) {
  return function(d) { return "Columns"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1369734149 = messageFormatterFn((function(  ) {
  return function(d) { return "Data Sources"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2117054896 = messageFormatterFn((function(  ) {
  return function(d) { return "Workbooks"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1432505507 = messageFormatterFn((function(  ) {
  return function(d) { return "Connects To"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2124845896 = messageFormatterFn((function(  ) {
  return function(d) { return "Data is"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a170012864 = messageFormatterFn((function(  ) {
  return function(d) { return "Connection Type"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a756810085 = messageFormatterFn((function(  ) {
  return function(d) { return "Connection Type"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a790531630 = messageFormatterFn((function(  ) {
  return function(d) { return "Contact"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1653025492 = messageFormatterFn((function(  ) {
  return function(d) { return "Sample"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1379881966 = messageFormatterFn((function(  ) {
  return function(d) { return "Database"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1640548470 = messageFormatterFn((function(  ) {
  return function(d) { return "Database name / Path"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a140852595 = messageFormatterFn((function(  ) {
  return function(d) { return "Database name / path"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a732990940 = messageFormatterFn((function(  ) {
  return function(d) { return "Data sources"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1780652662 = messageFormatterFn((function(  ) {
  return function(d) { return "Description"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b101529895 = messageFormatterFn((function(  ) {
  return function(d) { return "Live / Last extract"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a304079038 = messageFormatterFn((function(  ) {
  return function(d) { return "Live/Last Extract"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a138694589 = messageFormatterFn((function(  ) {
  return function(d) { return "Location"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a70149118 = messageFormatterFn((function(  ) {
  return function(d) { return "Metric Definitions"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a598817683 = messageFormatterFn((function(  ) {
  return function(d) { return "Name"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1536539377 = messageFormatterFn((function(  ) {
  return function(d) { return "Owned data sources"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1481339482 = messageFormatterFn((function(  ) {
  return function(d) { return "Owned metrics"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b758889187 = messageFormatterFn((function(  ) {
  return function(d) { return "Owned virtual connections"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1230418383 = messageFormatterFn((function(  ) {
  return function(d) { return "Owned virtual connection tables"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1963213924 = messageFormatterFn((function(  ) {
  return function(d) { return "Owned workbooks"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1393144639 = messageFormatterFn((function(  ) {
  return function(d) { return "Owner"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a306830055 = messageFormatterFn((function(  ) {
  return function(d) { return "Project"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b521796853 = messageFormatterFn((function(  ) {
  return function(d) { return "Server"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1015344595 = messageFormatterFn((function(  ) {
  return function(d) { return "Server / Path"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1753283196 = messageFormatterFn((function(  ) {
  return function(d) { return "Sheets"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b154573341 = messageFormatterFn((function(  ) {
  return function(d) { return "Source"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b250931338 = messageFormatterFn((function(  ) {
  return function(d) { return "Type"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a770427297 = messageFormatterFn((function(  ) {
  return function(d) { return "Tags"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1164818311 = messageFormatterFn((function(  ) {
  return function(d) { return "Total Followers"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a433317863 = messageFormatterFn((function(  ) {
  return function(d) { return "Metrics"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a792846466 = messageFormatterFn((function(  ) {
  return function(d) { return "Type"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a835018029 = messageFormatterFn((function(  ) {
  return function(d) { return "View"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a365809751 = messageFormatterFn((function(  ) {
  return function(d) { return "Virtual Connections"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1875286974 = messageFormatterFn((function(  ) {
  return function(d) { return "Workbook"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1995722055 = messageFormatterFn((function(  ) {
  return function(d) { return "Workbooks"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1966971047 = messageFormatterFn((function(  ) {
  return function(d) { return "Grouped asset"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1283157996 = messageFormatterFn((function(  ) {
  return function(d) { return d.name + " (grouped)"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a290831160 = messageFormatterFn((function(  ) {
  return function(d) { return "Grouped fields"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b265456438 = messageFormatterFn((function(  ) {
  return function(d) { return "Hidden User"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b151371928 = messageFormatterFn((function(  ) {
  return function(d) { return "Hostname"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1731091735 = messageFormatterFn((function(  ) {
  return function(d) { return "Inherited Description"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b839420742 = messageFormatterFn((function(  ) {
  return function(d) { return "On this asset"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b502398238 = messageFormatterFn((function(  ) {
  return function(d) { return "On this column"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1605599911 = messageFormatterFn((function( plural, pluralFuncs ) {
  return function(d) { return plural(d.MESSAGE_COUNT, 0, pluralFuncs.en, { one: function() { return "On column";}, other: function() { return "On columns";} }); }
})(messageFormat.plural, {"en": Globalize("en").pluralGenerator()}), Globalize("en").pluralGenerator({}));
Globalize.a910048234 = messageFormatterFn((function(  ) {
  return function(d) { return "On this connection"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2027514759 = messageFormatterFn((function(  ) {
  return function(d) { return "On this database"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1995861905 = messageFormatterFn((function(  ) {
  return function(d) { return "On this data source"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1553069318 = messageFormatterFn((function(  ) {
  return function(d) { return "On this flow"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1687550440 = messageFormatterFn((function(  ) {
  return function(d) { return "On this table"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a212590563 = messageFormatterFn((function(  ) {
  return function(d) { return "On this calculated insight"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a359443022 = messageFormatterFn((function(  ) {
  return function(d) { return "On this data lake"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b460251680 = messageFormatterFn((function(  ) {
  return function(d) { return "On this data model"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a383243585 = messageFormatterFn((function(  ) {
  return function(d) { return "Inherited by this asset"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1254509765 = messageFormatterFn((function( plural, pluralFuncs ) {
  return function(d) { return plural(d.MESSAGE_COUNT, 0, pluralFuncs.en, { one: function() { return "Inherited by column";}, other: function() { return "On or inherited by columns";} }); }
})(messageFormat.plural, {"en": Globalize("en").pluralGenerator()}), Globalize("en").pluralGenerator({}));
Globalize.a821565379 = messageFormatterFn((function(  ) {
  return function(d) { return "Inherited by this connection"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a802843040 = messageFormatterFn((function(  ) {
  return function(d) { return "Inherited by this database"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2002571964 = messageFormatterFn((function(  ) {
  return function(d) { return "Inherited by this calculated insight"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1049325397 = messageFormatterFn((function(  ) {
  return function(d) { return "Inherited by this data lake"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b548734535 = messageFormatterFn((function(  ) {
  return function(d) { return "Inherited by this data model"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1907379050 = messageFormatterFn((function(  ) {
  return function(d) { return "Inherited by this data source"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a226767051 = messageFormatterFn((function( plural, pluralFuncs ) {
  return function(d) { return plural(d.MESSAGE_COUNT, 0, pluralFuncs.en, { one: function() { return "Inherited by field";}, other: function() { return "Inherited by fields";} }); }
})(messageFormat.plural, {"en": Globalize("en").pluralGenerator()}), Globalize("en").pluralGenerator({}));
Globalize.b1236533869 = messageFormatterFn((function(  ) {
  return function(d) { return "Inherited by this flow"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1065244907 = messageFormatterFn((function(  ) {
  return function(d) { return "Inherited by this metric"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b464886113 = messageFormatterFn((function(  ) {
  return function(d) { return "Inherited by this table"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b781535926 = messageFormatterFn((function(  ) {
  return function(d) { return "Inherited by this view"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1358207521 = messageFormatterFn((function(  ) {
  return function(d) { return "Inherited by this workbook"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a745639543 = messageFormatterFn((function( plural, pluralFuncs ) {
  return function(d) { return plural(d.MESSAGE_COUNT, 0, pluralFuncs.en, { one: function() { return "Quality Warning";}, other: function() { return "Quality Warning (" + d.MESSAGE_COUNT_LOC + ")";} }); }
})(messageFormat.plural, {"en": Globalize("en").pluralGenerator()}), Globalize("en").pluralGenerator({}));
Globalize.b1822751660 = messageFormatterFn((function( plural, pluralFuncs ) {
  return function(d) { return plural(d.MESSAGE_COUNT, 0, pluralFuncs.en, { one: function() { return "Sensitivity";}, other: function() { return "Sensitivity (" + d.MESSAGE_COUNT_LOC + ")";} }); }
})(messageFormat.plural, {"en": Globalize("en").pluralGenerator()}), Globalize("en").pluralGenerator({}));
Globalize.b556217937 = messageFormatterFn((function( plural, pluralFuncs ) {
  return function(d) { return plural(d.MESSAGE_COUNT, 0, pluralFuncs.en, { one: function() { return d.LABEL_TYPE_LOC;}, other: function() { return d.LABEL_TYPE_LOC + " (" + d.MESSAGE_COUNT_LOC + ")";} }); }
})(messageFormat.plural, {"en": Globalize("en").pluralGenerator()}), Globalize("en").pluralGenerator({}));
Globalize.a1360998252 = messageFormatterFn((function(  ) {
  return function(d) { return "Upstream from this asset"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1008886160 = messageFormatterFn((function(  ) {
  return function(d) { return "Upstream from this column"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a681140344 = messageFormatterFn((function(  ) {
  return function(d) { return "Upstream from this connection"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a623926165 = messageFormatterFn((function(  ) {
  return function(d) { return "Upstream from this database"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1766954015 = messageFormatterFn((function(  ) {
  return function(d) { return "Upstream from this data source"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1204521718 = messageFormatterFn((function(  ) {
  return function(d) { return "Upstream from this field"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1343540728 = messageFormatterFn((function(  ) {
  return function(d) { return "Upstream from this flow"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b819621302 = messageFormatterFn((function(  ) {
  return function(d) { return "Upstream from this metric"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a512868554 = messageFormatterFn((function(  ) {
  return function(d) { return "Upstream from this table"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b888542785 = messageFormatterFn((function(  ) {
  return function(d) { return "Upstream from this view"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1537124396 = messageFormatterFn((function(  ) {
  return function(d) { return "Upstream from this workbook"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b396498058 = messageFormatterFn((function(  ) {
  return function(d) { return "View Lineage"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1454633942 = messageFormatterFn((function(  ) {
  return function(d) { return "Data Labels"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1403496434 = messageFormatterFn((function(  ) {
  return function(d) { return "Add New Category"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b313409056 = messageFormatterFn((function(  ) {
  return function(d) { return "Built-in Categories"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1820774189 = messageFormatterFn((function(  ) {
  return function(d) { return "Label category"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a650569943 = messageFormatterFn((function(  ) {
  return function(d) { return "Similar category name already exists"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b799930234 = messageFormatterFn((function(  ) {
  return function(d) { return "You can't change the category of a built-in label."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1184975339 = messageFormatterFn((function(  ) {
  return function(d) { return d.Category + " created. There may be a delay before you see your changes"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1362590978 = messageFormatterFn((function(  ) {
  return function(d) { return d.Category + " updated. There may be a delay before you see your changes"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a904386047 = messageFormatterFn((function(  ) {
  return function(d) { return "Category description (required)"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2028493785 = messageFormatterFn((function(  ) {
  return function(d) { return "Category description"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1587794922 = messageFormatterFn((function(  ) {
  return function(d) { return "Learn more"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a712979646 = messageFormatterFn((function(  ) {
  return function(d) { return "Category name"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b496315378 = messageFormatterFn((function(  ) {
  return function(d) { return "Data labels with custom categories don't appear on downstream assets. " + d.learnMoreLink; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1968471152 = messageFormatterFn((function(  ) {
  return function(d) { return "Sensitivity labels also appear on downstream assets. " + d.learnMoreLink; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a754332083 = messageFormatterFn((function(  ) {
  return function(d) { return "Data quality warnings also appear on downstream assets. " + d.learnMoreLink; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b72372606 = messageFormatterFn((function(  ) {
  return function(d) { return "Custom Categories"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1255007139 = messageFormatterFn((function(  ) {
  return function(d) { return "Label description (required)"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a320412319 = messageFormatterFn((function(  ) {
  return function(d) { return "New Category"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2119370371 = messageFormatterFn((function(  ) {
  return function(d) { return "Edit Category"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a532835211 = messageFormatterFn((function(  ) {
  return function(d) { return "Failed to retrieve data. Lineage information may still be loading. Try again later."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a41005788 = messageFormatterFn((function(  ) {
  return function(d) { return "Show Less"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a78826904 = messageFormatterFn((function(  ) {
  return function(d) { return "Show More"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1146468743 = messageFormatterFn((function(  ) {
  return function(d) { return "+ " + d.MESSAGE_COUNT; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a154223890 = messageFormatterFn((function(  ) {
  return function(d) { return "Label value"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1520176392 = messageFormatterFn((function(  ) {
  return function(d) { return "Similar label name already exists"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b17297657 = messageFormatterFn((function(  ) {
  return function(d) { return "This asset is trusted and recommended."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1714110260 = messageFormatterFn((function(  ) {
  return function(d) { return d.Category + " label " + d.name + " created. There may be a delay before you see your changes"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1991510573 = messageFormatterFn((function(  ) {
  return function(d) { return "This asset is no longer maintained and shouldn't be used."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1231796314 = messageFormatterFn((function(  ) {
  return function(d) { return "This asset's most recent extract refresh failed."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b334827791 = messageFormatterFn((function(  ) {
  return function(d) { return "This flow's most recent run failed."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2097260901 = messageFormatterFn((function(  ) {
  return function(d) { return "This asset is undergoing maintenance."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1037826488 = messageFormatterFn((function(  ) {
  return function(d) { return "This asset contains sensitive information."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1442695915 = messageFormatterFn((function(  ) {
  return function(d) { return "This asset is outdated."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a33290719 = messageFormatterFn((function(  ) {
  return function(d) { return d.Category + " label " + d.name + " updated. There may be a delay before you see your changes"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1858120654 = messageFormatterFn((function(  ) {
  return function(d) { return "This asset has a general quality issue."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2129270961 = messageFormatterFn((function(  ) {
  return function(d) { return "High visibility"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b110064781 = messageFormatterFn((function(  ) {
  return function(d) { return "Set visibility level"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a980674308 = messageFormatterFn((function(  ) {
  return function(d) { return "Standard visibility"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b79033025 = messageFormatterFn((function(  ) {
  return function(d) { return "Last 30 days"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2031796615 = messageFormatterFn((function(  ) {
  return function(d) { return "Last 7 days"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1787894393 = messageFormatterFn((function(  ) {
  return function(d) { return "Last 90 days"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2069721752 = messageFormatterFn((function(  ) {
  return function(d) { return "Least-Most"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1306376799 = messageFormatterFn((function(  ) {
  return function(d) { return "Lenses"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1197620126 = messageFormatterFn((function(  ) {
  return function(d) { return "Lenses (" + d.count + ")"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b846703110 = messageFormatterFn((function(  ) {
  return function(d) { return "Lineage"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2108350774 = messageFormatterFn((function(  ) {
  return function(d) { return d.name + " (" + d.parentName + ")"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b834379041 = messageFormatterFn((function(  ) {
  return function(d) { return "Live"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b673214467 = messageFormatterFn((function(  ) {
  return function(d) { return "Loading…"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2111087407 = messageFormatterFn((function(  ) {
  return function(d) { return "Loading data details…"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1270116167 = messageFormatterFn((function(  ) {
  return function(d) { return "Retry"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1649256900 = messageFormatterFn((function(  ) {
  return function(d) { return "Metrics"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1378758949 = messageFormatterFn((function(  ) {
  return function(d) { return "Metrics (" + d.count + ")"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2117892184 = messageFormatterFn((function(  ) {
  return function(d) { return "Metric Definitions"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1897932487 = messageFormatterFn((function(  ) {
  return function(d) { return "Metric Definitions (" + d.count + ")"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1048928428 = messageFormatterFn((function(  ) {
  return function(d) { return "Search for data"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1394186366 = messageFormatterFn((function(  ) {
  return function(d) { return "Modified"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b876912601 = messageFormatterFn((function(  ) {
  return function(d) { return "More Filters"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1828885612 = messageFormatterFn((function(  ) {
  return function(d) { return "More Filters Applied"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b911395201 = messageFormatterFn((function(  ) {
  return function(d) { return "All Data Labels…"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2057645140 = messageFormatterFn((function(  ) {
  return function(d) { return "Most-Least"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1662214753 = messageFormatterFn((function(  ) {
  return function(d) { return "New"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1178700033 = messageFormatterFn((function(  ) {
  return function(d) { return "New Flow"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b567993221 = messageFormatterFn((function(  ) {
  return function(d) { return "New Label"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2110268851 = messageFormatterFn((function(  ) {
  return function(d) { return "New Workbook"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b989850493 = messageFormatterFn((function(  ) {
  return function(d) { return "No description available."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b952248377 = messageFormatterFn((function(  ) {
  return function(d) { return "No description"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1920467 = messageFormatterFn((function(  ) {
  return function(d) { return "none"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1347382243 = messageFormatterFn((function(  ) {
  return function(d) { return "None"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1614445737 = messageFormatterFn((function(  ) {
  return function(d) { return "This asset is not in a project or personal space."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b161690000 = messageFormatterFn((function(  ) {
  return function(d) { return "No tags available."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a159908755 = messageFormatterFn((function(  ) {
  return function(d) { return "Not Applicable"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1814868805 = messageFormatterFn((function(  ) {
  return function(d) { return "Embedded assets cannot be in a project or personal space."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2084567439 = messageFormatterFn((function( plural, pluralFuncs ) {
  return function(d) { return plural(d.USER_COUNT, 0, pluralFuncs.en, { one: function() { return d.USER_COUNT_LOC + " recipient";}, other: function() { return d.USER_COUNT_LOC + " recipients";} }); }
})(messageFormat.plural, {"en": Globalize("en").pluralGenerator()}), Globalize("en").pluralGenerator({}));
Globalize.a965648977 = messageFormatterFn((function(  ) {
  return function(d) { return "Other"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a77409255 = messageFormatterFn((function(  ) {
  return function(d) { return "Output Steps"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b438369256 = messageFormatterFn((function(  ) {
  return function(d) { return "Output Steps (" + d.count + ")"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1057077556 = messageFormatterFn((function(  ) {
  return function(d) { return "Owner"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a943026479 = messageFormatterFn((function(  ) {
  return function(d) { return "Owner's Display Name"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2054548113 = messageFormatterFn((function(  ) {
  return function(d) { return "Owner's Id"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1590258027 = messageFormatterFn((function(  ) {
  return function(d) { return "Owners"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b668484310 = messageFormatterFn((function(  ) {
  return function(d) { return "Owners (" + d.count + ")"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a135024026 = messageFormatterFn((function(  ) {
  return function(d) { return "(" + d.ITEM + ")"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1463271805 = messageFormatterFn((function(  ) {
  return function(d) { return "Permission required"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b888591462 = messageFormatterFn((function(  ) {
  return function(d) { return "This warning is on an asset you do not have permissions to see."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1505038404 = messageFormatterFn((function(  ) {
  return function(d) { return "Permissions Required"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2102999139 = messageFormatterFn((function(  ) {
  return function(d) { return "Personal Space"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1591888978 = messageFormatterFn((function(  ) {
  return function(d) { return "Personal Space"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1070349380 = messageFormatterFn((function(  ) {
  return function(d) { return "+ " + d.count; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b184698779 = messageFormatterFn((function(  ) {
  return function(d) { return " +" + d.count; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1618362404 = messageFormatterFn((function(  ) {
  return function(d) { return "Popular Data"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1328281242 = messageFormatterFn((function(  ) {
  return function(d) { return "Project"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1606697211 = messageFormatterFn((function(  ) {
  return function(d) { return "Project Name"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1554519341 = messageFormatterFn((function(  ) {
  return function(d) { return "Virtual Connection"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1276577706 = messageFormatterFn((function(  ) {
  return function(d) { return "Virtual Connection (Extract)"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1000046137 = messageFormatterFn((function(  ) {
  return function(d) { return "Virtual Connection (Live)"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a945535420 = messageFormatterFn((function(  ) {
  return function(d) { return "Virtual Connections"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1135987229 = messageFormatterFn((function(  ) {
  return function(d) { return "Virtual Connections (" + d.count + ")"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a109865326 = messageFormatterFn((function(  ) {
  return function(d) { return "Query metadata (GraphiQL)"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b479519936 = messageFormatterFn((function(  ) {
  return function(d) { return "read more"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a669205511 = messageFormatterFn((function(  ) {
  return function(d) { return "Referenced fields"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1316122278 = messageFormatterFn((function(  ) {
  return function(d) { return "Relevance"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a528720377 = messageFormatterFn((function(  ) {
  return function(d) { return "Remove"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2048844887 = messageFormatterFn((function(  ) {
  return function(d) { return "Boolean"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b700237469 = messageFormatterFn((function(  ) {
  return function(d) { return "Date"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2074101008 = messageFormatterFn((function(  ) {
  return function(d) { return "Date & Time"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a680149823 = messageFormatterFn((function(  ) {
  return function(d) { return "Integer"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b296294573 = messageFormatterFn((function(  ) {
  return function(d) { return "Real"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1385420422 = messageFormatterFn((function(  ) {
  return function(d) { return "String"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1333517131 = messageFormatterFn((function(  ) {
  return function(d) { return "Unknown"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2127197528 = messageFormatterFn((function(  ) {
  return function(d) { return "Reset Filter"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1631242671 = messageFormatterFn((function(  ) {
  return function(d) { return "Revert"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1050260032 = messageFormatterFn((function(  ) {
  return function(d) { return "Reverting will return the built-in label to its defaults, and will affect existing instances of the label that are attached to assets. Are you sure you want to revert the label to its defaults?"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a702289023 = messageFormatterFn((function(  ) {
  return function(d) { return "Revert to default label"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a305171907 = messageFormatterFn((function(  ) {
  return function(d) { return d.LABEL_CATEGORY + " label " + d.LABEL_VALUE + " reverted to defaults. There might be a delay before you see your changes."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a128534322 = messageFormatterFn((function(  ) {
  return function(d) { return "Save"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a343870422 = messageFormatterFn((function(  ) {
  return function(d) { return "Schema"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1470826474 = messageFormatterFn((function(  ) {
  return function(d) { return "Search for data"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1088655853 = messageFormatterFn((function(  ) {
  return function(d) { return "Search"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1310928623 = messageFormatterFn((function(  ) {
  return function(d) { return d.count + " results"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b555650775 = messageFormatterFn((function(  ) {
  return function(d) { return d.count + " results for "; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1999724355 = messageFormatterFn((function(  ) {
  return function(d) { return "See All"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2001863174 = messageFormatterFn((function(  ) {
  return function(d) { return "Select All"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a131989117 = messageFormatterFn((function(  ) {
  return function(d) { return "Send"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b87136203 = messageFormatterFn((function(  ) {
  return function(d) { return "Send Email"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1740373998 = messageFormatterFn((function(  ) {
  return function(d) { return "You don't have permissions to send email from here."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1925495846 = messageFormatterFn((function(  ) {
  return function(d) { return "Select users to send an email."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1073940023 = messageFormatterFn((function(  ) {
  return function(d) { return "Sensitivity Labels…"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1242795694 = messageFormatterFn((function(  ) {
  return function(d) { return "Sensitivity"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a424686464 = messageFormatterFn((function(  ) {
  return function(d) { return "Fields in Use (" + d.count + ")"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a400759913 = messageFormatterFn((function(  ) {
  return function(d) { return "Sheets"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1752005162 = messageFormatterFn((function(  ) {
  return function(d) { return "Sheets (" + d.count + ")"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a291843621 = messageFormatterFn((function(  ) {
  return function(d) { return "Show All"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a765958315 = messageFormatterFn((function(  ) {
  return function(d) { return "show less"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1867223749 = messageFormatterFn((function(  ) {
  return function(d) { return "Connection Type (a–z)"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2122856915 = messageFormatterFn((function(  ) {
  return function(d) { return "Connection Type (z–a)"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1949739237 = messageFormatterFn((function(  ) {
  return function(d) { return "Contact (a–z)"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b238878659 = messageFormatterFn((function(  ) {
  return function(d) { return "Contact (z–a)"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b915810225 = messageFormatterFn((function(  ) {
  return function(d) { return "Database name / Path (a-z)"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1748149641 = messageFormatterFn((function(  ) {
  return function(d) { return "Database name / Path (z-a)"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2086281145 = messageFormatterFn((function(  ) {
  return function(d) { return "Sheets (least-most)"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b176710511 = messageFormatterFn((function(  ) {
  return function(d) { return "Sheets (most-least)"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a428765724 = messageFormatterFn((function(  ) {
  return function(d) { return "Sort By:"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a105685623 = messageFormatterFn((function(  ) {
  return function(d) { return "Order"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b577152341 = messageFormatterFn((function(  ) {
  return function(d) { return "Sort"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1029619566 = messageFormatterFn((function(  ) {
  return function(d) { return "Name (a–z)"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1779939930 = messageFormatterFn((function(  ) {
  return function(d) { return "Name (z–a)"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1298067819 = messageFormatterFn((function(  ) {
  return function(d) { return "Data Sources (least–most)"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1511901181 = messageFormatterFn((function(  ) {
  return function(d) { return "Data Sources (most–least)"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1790725578 = messageFormatterFn((function(  ) {
  return function(d) { return "Workbooks (least–most)"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b248586386 = messageFormatterFn((function(  ) {
  return function(d) { return "Workbooks (most–least)"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b276613273 = messageFormatterFn((function(  ) {
  return function(d) { return "A–Z"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b275892523 = messageFormatterFn((function(  ) {
  return function(d) { return "Z–A"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1194448904 = messageFormatterFn((function(  ) {
  return function(d) { return "Least–Most"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a534492322 = messageFormatterFn((function(  ) {
  return function(d) { return "Most–Least"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2018592543 = messageFormatterFn((function(  ) {
  return function(d) { return "Popularity"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a91438853 = messageFormatterFn((function(  ) {
  return function(d) { return "Relevance"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1254036875 = messageFormatterFn((function(  ) {
  return function(d) { return "Server / Path (a-z)"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b146941917 = messageFormatterFn((function(  ) {
  return function(d) { return "Server / Path (z-a)"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1291218586 = messageFormatterFn((function(  ) {
  return function(d) { return "Source Data"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2019999478 = messageFormatterFn((function(  ) {
  return function(d) { return "Views over last 12 weeks"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1025420147 = messageFormatterFn((function(  ) {
  return function(d) { return "Submit"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a558901135 = messageFormatterFn((function(  ) {
  return function(d) { return "Table"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1320701949 = messageFormatterFn((function(  ) {
  return function(d) { return "Table (" + d.name + ")"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a146142106 = messageFormatterFn((function(  ) {
  return function(d) { return "Tables"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1858192631 = messageFormatterFn((function(  ) {
  return function(d) { return "Tables and Objects"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b485476923 = messageFormatterFn((function(  ) {
  return function(d) { return "Tables (" + d.count + ")"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b848454392 = messageFormatterFn((function(  ) {
  return function(d) { return "Tables and Objects (" + d.count + ")"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1667621339 = messageFormatterFn((function(  ) {
  return function(d) { return "Tag"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1136294223 = messageFormatterFn((function(  ) {
  return function(d) { return "Tag…"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a156730062 = messageFormatterFn((function(  ) {
  return function(d) { return "Tags"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a330900272 = messageFormatterFn((function(  ) {
  return function(d) { return "To"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a961247810 = messageFormatterFn((function(  ) {
  return function(d) { return "Today"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1855278587 = messageFormatterFn((function(  ) {
  return function(d) { return "Toggle filters"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1923202784 = messageFormatterFn((function(  ) {
  return function(d) { return "(" + d.count + " total)"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a296218049 = messageFormatterFn((function(  ) {
  return function(d) { return "The URL is not recognized."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b696264258 = messageFormatterFn((function(  ) {
  return function(d) { return "Upstream Databases"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b581109727 = messageFormatterFn((function(  ) {
  return function(d) { return "Upstream Databases (" + d.count + ")"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1006960628 = messageFormatterFn((function(  ) {
  return function(d) { return "Upstream Published Data Sources"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1270902443 = messageFormatterFn((function(  ) {
  return function(d) { return "Upstream Published Data Sources (" + d.count + ")"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b85387605 = messageFormatterFn((function(  ) {
  return function(d) { return "Upstream Flows"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b40569388 = messageFormatterFn((function(  ) {
  return function(d) { return "Upstream Flows (" + d.count + ")"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1564656284 = messageFormatterFn((function(  ) {
  return function(d) { return "Upstream Lenses"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a923740419 = messageFormatterFn((function(  ) {
  return function(d) { return "Upstream Lenses (" + d.count + ")"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1277792107 = messageFormatterFn((function(  ) {
  return function(d) { return "Upstream Tables"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1957856306 = messageFormatterFn((function(  ) {
  return function(d) { return "Upstream Tables and Objects"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b759356630 = messageFormatterFn((function(  ) {
  return function(d) { return "Upstream Tables (" + d.count + ")"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1915695059 = messageFormatterFn((function(  ) {
  return function(d) { return "Upstream Tables and Objects (" + d.count + ")"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1345382074 = messageFormatterFn((function(  ) {
  return function(d) { return "Upstream Virtual Connections"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1536899419 = messageFormatterFn((function(  ) {
  return function(d) { return "Upstream Virtual Connections (" + d.count + ")"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2065553740 = messageFormatterFn((function(  ) {
  return function(d) { return "Upstream Virtual Connection Tables"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b916267029 = messageFormatterFn((function(  ) {
  return function(d) { return "Upstream Virtual Connection Tables (" + d.count + ")"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1026995330 = messageFormatterFn((function(  ) {
  return function(d) { return "Used by fields"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a544233202 = messageFormatterFn((function(  ) {
  return function(d) { return "Used by flows"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a221320794 = messageFormatterFn((function(  ) {
  return function(d) { return "View"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a622155645 = messageFormatterFn((function(  ) {
  return function(d) { return "View all"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1490266282 = messageFormatterFn((function(  ) {
  return function(d) { return "Views of this dashboard"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1259726283 = messageFormatterFn((function(  ) {
  return function(d) { return "Views of this sheet"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1326436739 = messageFormatterFn((function(  ) {
  return function(d) { return "View Description"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1879270829 = messageFormatterFn((function(  ) {
  return function(d) { return "View less"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1841449713 = messageFormatterFn((function(  ) {
  return function(d) { return "View More"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1525885002 = messageFormatterFn((function(  ) {
  return function(d) { return "Virtual Connection"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a57870911 = messageFormatterFn((function(  ) {
  return function(d) { return "Virtual Connections"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1879165120 = messageFormatterFn((function(  ) {
  return function(d) { return "Virtual Connections (" + d.count + ")"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1502674374 = messageFormatterFn((function(  ) {
  return function(d) { return "Virtual Connection Table"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a661810767 = messageFormatterFn((function(  ) {
  return function(d) { return "Virtual Connection Tables"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1336538800 = messageFormatterFn((function(  ) {
  return function(d) { return "Virtual Connection Tables (" + d.count + ")"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1766808828 = messageFormatterFn((function(  ) {
  return function(d) { return "Web Data Connector"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b129752351 = messageFormatterFn((function(  ) {
  return function(d) { return "Web Data Connector URL"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b275071249 = messageFormatterFn((function(  ) {
  return function(d) { return "Workbook"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b791812082 = messageFormatterFn((function(  ) {
  return function(d) { return "Workbook"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1945671786 = messageFormatterFn((function(  ) {
  return function(d) { return "Workbook Id"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a62801978 = messageFormatterFn((function(  ) {
  return function(d) { return "Workbooks"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1089829686 = messageFormatterFn((function(  ) {
  return function(d) { return "Workbook Using This Database"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a344490099 = messageFormatterFn((function(  ) {
  return function(d) { return "Workbook Using This Table"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1743973083 = messageFormatterFn((function(  ) {
  return function(d) { return "Workbooks (" + d.count + ")"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1150316586 = messageFormatterFn((function( plural, pluralFuncs ) {
  return function(d) { return plural(d.COUNT, 0, pluralFuncs.en, { one: function() { return d.COUNT_LOC + " column selected";}, other: function() { return d.COUNT_LOC + " columns selected";} }); }
})(messageFormat.plural, {"en": Globalize("en").pluralGenerator()}), Globalize("en").pluralGenerator({}));
Globalize.b1613128366 = messageFormatterFn((function( plural, pluralFuncs ) {
  return function(d) { return plural(d.COUNT, 0, pluralFuncs.en, { one: function() { return d.COUNT_LOC + " field selected";}, other: function() { return d.COUNT_LOC + " fields selected";} }); }
})(messageFormat.plural, {"en": Globalize("en").pluralGenerator()}), Globalize("en").pluralGenerator({}));
Globalize.a587965592 = messageFormatterFn((function( plural, pluralFuncs ) {
  return function(d) { return plural(d.COUNT, 0, pluralFuncs.en, { one: function() { return d.COUNT_LOC + " item selected";}, other: function() { return d.COUNT_LOC + " items selected";} }); }
})(messageFormat.plural, {"en": Globalize("en").pluralGenerator()}), Globalize("en").pluralGenerator({}));
Globalize.a284110612 = messageFormatterFn((function(  ) {
  return function(d) { return d.countX + " of " + d.countN; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a180809538 = messageFormatterFn((function(  ) {
  return function(d) { return d.countX + "/" + d.countN; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1145487142 = messageFormatterFn((function( plural, pluralFuncs ) {
  return function(d) { return plural(d.COUNT, 0, pluralFuncs.en, { one: function() { return d.COUNT_LOC + " table selected";}, other: function() { return d.COUNT_LOC + " tables selected";} }); }
})(messageFormat.plural, {"en": Globalize("en").pluralGenerator()}), Globalize("en").pluralGenerator({}));
Globalize.a341018295 = messageFormatterFn((function( plural, pluralFuncs ) {
  return function(d) { return plural(d.COUNT, 0, pluralFuncs.en, { one: function() { return d.COUNT_LOC + " data model selected";}, other: function() { return d.COUNT_LOC + " data models selected";} }); }
})(messageFormat.plural, {"en": Globalize("en").pluralGenerator()}), Globalize("en").pluralGenerator({}));
Globalize.b2130048701 = messageFormatterFn((function(  ) {
  return function(d) { return "Yesterday"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a498465828 = messageFormatterFn((function(  ) {
  return function(d) { return "Show template project contributors!"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b110550615 = messageFormatterFn((function(  ) {
  return function(d) { return "Hide template project contributors!"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a753393564 = messageFormatterFn((function(  ) {
  return function(d) { return "Current Analytics Connection"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1202461751 = messageFormatterFn((function(  ) {
  return function(d) { return "Select a Connection"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b568082692 = messageFormatterFn((function(  ) {
  return function(d) { return "Choose a Connection"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b882607199 = messageFormatterFn((function(  ) {
  return function(d) { return "No Connection"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a916463610 = messageFormatterFn((function(  ) {
  return function(d) { return "No analytics extension configured for this workbook"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1782175721 = messageFormatterFn((function(  ) {
  return function(d) { return "Script"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b35735768 = messageFormatterFn((function(  ) {
  return function(d) { return "Enter Custom Script"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a268693453 = messageFormatterFn((function(  ) {
  return function(d) { return "Apply"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b479702763 = messageFormatterFn((function(  ) {
  return function(d) { return "Model/Prediction ID"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1112389892 = messageFormatterFn((function(  ) {
  return function(d) { return "Show advanced parameters"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1982815934 = messageFormatterFn((function(  ) {
  return function(d) { return "Max Middle Values"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2110585360 = messageFormatterFn((function(  ) {
  return function(d) { return "Max Prescriptions"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1004098679 = messageFormatterFn((function(  ) {
  return function(d) { return "Show Multi-Class Probabilities"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a744925837 = messageFormatterFn((function(  ) {
  return function(d) { return "See Documentation"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b532127822 = messageFormatterFn((function(  ) {
  return function(d) { return "1 null not shown"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b707348513 = messageFormatterFn((function(  ) {
  return function(d) { return d.number + " nulls not shown"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1350078579 = messageFormatterFn((function(  ) {
  return function(d) { return "Zoom in"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1990168310 = messageFormatterFn((function(  ) {
  return function(d) { return "Reset axes"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a926083224 = messageFormatterFn((function(  ) {
  return function(d) { return "Format"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1675931001 = messageFormatterFn((function(  ) {
  return function(d) { return "Uh-oh something went wrong! Please close the formatting pane and try that again."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1675930040 = messageFormatterFn((function(  ) {
  return function(d) { return "We've sent an error report to our engineers."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1584655079 = messageFormatterFn((function(  ) {
  return function(d) { return "None"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a780811780 = messageFormatterFn((function(  ) {
  return function(d) { return "Off"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1819282268 = messageFormatterFn((function(  ) {
  return function(d) { return "Fill"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a774334640 = messageFormatterFn((function(  ) {
  return function(d) { return "HexColor"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1850512113 = messageFormatterFn((function(  ) {
  return function(d) { return "Multiple"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2130840374 = messageFormatterFn((function(  ) {
  return function(d) { return "Font Family"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b652983623 = messageFormatterFn((function(  ) {
  return function(d) { return "Font Size"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2121653134 = messageFormatterFn((function(  ) {
  return function(d) { return "Color"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1485680619 = messageFormatterFn((function(  ) {
  return function(d) { return "Worksheet Title"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a922592146 = messageFormatterFn((function(  ) {
  return function(d) { return "Decimal Places"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a5167492 = messageFormatterFn((function(  ) {
  return function(d) { return "Units"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b434702150 = messageFormatterFn((function(  ) {
  return function(d) { return "Automatic"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1997931450 = messageFormatterFn((function(  ) {
  return function(d) { return "On"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1806507522 = messageFormatterFn((function(  ) {
  return function(d) { return "Off"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1738782272 = messageFormatterFn((function(  ) {
  return function(d) { return "Solid"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1799256587 = messageFormatterFn((function(  ) {
  return function(d) { return "Dotted"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2075571758 = messageFormatterFn((function(  ) {
  return function(d) { return "Dashed"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1382257603 = messageFormatterFn((function(  ) {
  return function(d) { return "Line Width"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b911215240 = messageFormatterFn((function(  ) {
  return function(d) { return "Line Style"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a49817456 = messageFormatterFn((function(  ) {
  return function(d) { return "Show title"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1831797354 = messageFormatterFn((function(  ) {
  return function(d) { return "Unable to load microview from " + d.MICROVIEW_NAME + "."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b275519392 = messageFormatterFn((function(  ) {
  return function(d) { return "Something went wrong, and we couldn’t complete your search. Try again later."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1983738672 = messageFormatterFn((function(  ) {
  return function(d) { return "Check Best Practices"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a382676720 = messageFormatterFn((function(  ) {
  return function(d) { return "Select an item to see the best practices guidelines and information on resolving issues."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1472372142 = messageFormatterFn((function(  ) {
  return function(d) { return "Updating these items to follow best practices won't impact workbook functionality."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a408220097 = messageFormatterFn((function(  ) {
  return function(d) { return "Updating these items to follow best practices may require a trade-off in the workbook."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b727839839 = messageFormatterFn((function(  ) {
  return function(d) { return "Passed items follow best practice guidelines."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1649636277 = messageFormatterFn((function(  ) {
  return function(d) { return "Items that have been ignored."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b900977551 = messageFormatterFn((function(  ) {
  return function(d) { return "Passed"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b771263993 = messageFormatterFn((function(  ) {
  return function(d) { return "Ignored"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a96475744 = messageFormatterFn((function(  ) {
  return function(d) { return "Take action"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1340594769 = messageFormatterFn((function(  ) {
  return function(d) { return "Needs review"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1875962168 = messageFormatterFn((function(  ) {
  return function(d) { return d.passedRules + "/" + d.totalRules; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1934524178 = messageFormatterFn((function(  ) {
  return function(d) { return "…"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a74267516 = messageFormatterFn((function(  ) {
  return function(d) { return ", "; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1852909264 = messageFormatterFn((function(  ) {
  return function(d) { return "+" + d.numExceeding + " more"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1010322594 = messageFormatterFn((function(  ) {
  return function(d) { return "Rerun Optimizer"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a232077060 = messageFormatterFn((function(  ) {
  return function(d) { return "Publish"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a521136173 = messageFormatterFn((function(  ) {
  return function(d) { return "Close"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b127243090 = messageFormatterFn((function(  ) {
  return function(d) { return "Learn More"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a786010250 = messageFormatterFn((function(  ) {
  return function(d) { return "Last run at " + d.time; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1037332724 = messageFormatterFn((function(  ) {
  return function(d) { return "Scanning Workbook"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1210037553 = messageFormatterFn((function(  ) {
  return function(d) { return "The workbook follows best practices for this guideline."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b94031390 = messageFormatterFn((function(  ) {
  return function(d) { return "The workbook does not follow best practices for this guideline."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1365858378 = messageFormatterFn((function(  ) {
  return function(d) { return "Copy"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b316109945 = messageFormatterFn((function(  ) {
  return function(d) { return "Ignore this rule"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1553506610 = messageFormatterFn((function(  ) {
  return function(d) { return "Evaluate this rule"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1754934433 = messageFormatterFn((function(  ) {
  return function(d) { return "Fix the rule"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a761411634 = messageFormatterFn((function(  ) {
  return function(d) { return "Fix"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a799479960 = messageFormatterFn((function(  ) {
  return function(d) { return "Ignore rule"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a840013593 = messageFormatterFn((function(  ) {
  return function(d) { return d.ruleName + " was fixed."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b659285886 = messageFormatterFn((function(  ) {
  return function(d) { return " Refresh the optimizer to see these changes."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1304830495 = messageFormatterFn((function(  ) {
  return function(d) { return "The rule wasn't fixed. Try resolving the issue manually."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b395636658 = messageFormatterFn((function(  ) {
  return function(d) { return d.ruleName + " won't be evaluated again for this workbook. Refresh the optimizer to see this change."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b468385707 = messageFormatterFn((function(  ) {
  return function(d) { return d.ruleName + " will be evaluated for this workbook. Refresh the optimizer to see this change."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b921497985 = messageFormatterFn((function(  ) {
  return function(d) { return "Try clicking here and holding " + d.shortcutHint + "."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2127957205 = messageFormatterFn((function(  ) {
  return function(d) { return "Number of data sources"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1351598724 = messageFormatterFn((function(  ) {
  return function(d) { return "The workbook contains " + d.numDatasources + " data sources."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2030247596 = messageFormatterFn((function(  ) {
  return function(d) { return "The workbook follows best practices with few data sources."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1177252594 = messageFormatterFn((function(  ) {
  return function(d) { return "Each data source contributes to the amount of time Tableau spends loading and rendering a workbook. Consider combining data sources if possible, especially when the level of granularity is the same or they support the same analysis."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b83175517 = messageFormatterFn((function(  ) {
  return function(d) { return "Number of visible workbook sheets"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1019011076 = messageFormatterFn((function(  ) {
  return function(d) { return "The workbook contains " + d.numVisibleWindows + " visible sheets."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1457669324 = messageFormatterFn((function(  ) {
  return function(d) { return "The workbook follows best practices with limited visible sheets."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a768546682 = messageFormatterFn((function(  ) {
  return function(d) { return "The overall size of a workbook impacts how long it takes to be processed and displayed. Reducing the number of sheets can help speed up load time. Consider reducing the number of sheets in the workbook by closing unneeded sheets or splitting the analysis into multiple workbooks."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b850683835 = messageFormatterFn((function(  ) {
  return function(d) { return "Number of hidden workbook sheets"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1647463006 = messageFormatterFn((function(  ) {
  return function(d) { return "The workbook contains " + d.numHiddenWindows + " hidden sheets."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a865296978 = messageFormatterFn((function(  ) {
  return function(d) { return "The workbook follows best practices with limited hidden sheets."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b239813480 = messageFormatterFn((function(  ) {
  return function(d) { return "Hidden sheets add to overall workbook complexity. Consider reducing the total number of sheets in the workbook by removing items from dashboards or separating dashboards into multiple workbooks."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1438519793 = messageFormatterFn((function(  ) {
  return function(d) { return "Number of views in a dashboard"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a437928728 = messageFormatterFn((function(  ) {
  return function(d) { return "Dashboard " + d.dashboardName + " contains " + d.numSheets + " views."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1557294008 = messageFormatterFn((function(  ) {
  return function(d) { return "Multiple dashboards contain more than 10 views."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2007072584 = messageFormatterFn((function(  ) {
  return function(d) { return "The workbook follows best practices with limited views per dashboard."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1871851406 = messageFormatterFn((function(  ) {
  return function(d) { return "A dashboard must load all elements before it can be displayed. The more views a dashboard contains, the longer it will take to load. Consider streamlining the dashboard to be as effective and simple as possible by reducing the number of views, filters, and other elements."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1547515683 = messageFormatterFn((function(  ) {
  return function(d) { return "Calculation uses multiple data sources"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b301486070 = messageFormatterFn((function( plural, pluralFuncs ) {
  return function(d) { return plural(d.itemCount, 0, pluralFuncs.en, { one: function() { return "Data source " + d.datasourceName + " has " + d.numCalcs + " calculation using another data source.";}, other: function() { return "Data source " + d.datasourceName + " has " + d.numCalcs + " calculations using another data source.";} }); }
})(messageFormat.plural, {"en": Globalize("en").pluralGenerator()}), Globalize("en").pluralGenerator({}));
Globalize.a1101000506 = messageFormatterFn((function(  ) {
  return function(d) { return "Multiple data sources have calculations with fields from other data sources."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a141684410 = messageFormatterFn((function(  ) {
  return function(d) { return "The workbook follows best practices and no calcs in use span data sources."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a494547200 = messageFormatterFn((function(  ) {
  return function(d) { return "When calculations use fields from multiple data sources, Tableau can't leverage any optimizations and must compute the calculation locally. Consider using Tableau Prep to create the calculation and using an extract."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1178514802 = messageFormatterFn((function(  ) {
  return function(d) { return "Unused fields"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b923526379 = messageFormatterFn((function( plural, pluralFuncs ) {
  return function(d) { return plural(d.itemCount, 0, pluralFuncs.en, { one: function() { return "The data source " + d.datasourceName + " contains " + d.numUnusedFields + " field not being used.";}, other: function() { return "The data source " + d.datasourceName + " contains " + d.numUnusedFields + " fields not being used.";} }); }
})(messageFormat.plural, {"en": Globalize("en").pluralGenerator()}), Globalize("en").pluralGenerator({}));
Globalize.b2096716411 = messageFormatterFn((function(  ) {
  return function(d) { return "The workbook follows best practices and has no unused fields."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1776426821 = messageFormatterFn((function(  ) {
  return function(d) { return "Multiple data sources have unused fields."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b703864117 = messageFormatterFn((function(  ) {
  return function(d) { return "Hiding unused fields will prevent them from being unnecessarily queried and reduce the size of extracts. Consider hiding any fields that are not being used, regardless of whether or not the data source is an extract."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1817936295 = messageFormatterFn((function(  ) {
  return function(d) { return "Unused field hidden."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1571618441 = messageFormatterFn((function(  ) {
  return function(d) { return "Unused fields hidden."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b287666280 = messageFormatterFn((function(  ) {
  return function(d) { return "Number of filters"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1811172271 = messageFormatterFn((function(  ) {
  return function(d) { return "The sheet " + d.sheetName + " uses " + d.numFilters + " interactive filters."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a316478943 = messageFormatterFn((function(  ) {
  return function(d) { return "Multiple sheets use more than 10 interactive filters."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b475891041 = messageFormatterFn((function(  ) {
  return function(d) { return "The workbook follows best practices using few interactive filters per view."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2068614683 = messageFormatterFn((function(  ) {
  return function(d) { return "Excessive filters on a view create a more complex query. Consider reducing the number of filters and using filter actions wherever possible. Not all types and formats of filters are created equal in terms of performance, so consider optimizing necessary filters."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1863739291 = messageFormatterFn((function(  ) {
  return function(d) { return "Multiple connections in a data source"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b764950772 = messageFormatterFn((function(  ) {
  return function(d) { return "The data source " + d.datasourceName + " uses multiple data connections."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a344056892 = messageFormatterFn((function(  ) {
  return function(d) { return "Multiple data sources use multiple connections."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1039219772 = messageFormatterFn((function(  ) {
  return function(d) { return "The workbook follows best practices and data sources use single connections."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1852301182 = messageFormatterFn((function(  ) {
  return function(d) { return "Data sources that include multiple connections cannot be computed locally. Consider combining the data sources in Tableau Prep and using an extract for analysis."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1012409016 = messageFormatterFn((function(  ) {
  return function(d) { return "Number of LOD calculations"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1219795969 = messageFormatterFn((function(  ) {
  return function(d) { return "The data source " + d.datasourceName + " contains " + d.numLodCalculations + " LOD calculations."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b687940049 = messageFormatterFn((function(  ) {
  return function(d) { return "Multiple data sources contain more than 5 LOD calculations."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a792336111 = messageFormatterFn((function(  ) {
  return function(d) { return "The workbook follows best practices and no data sources use excessive LODs."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1150574133 = messageFormatterFn((function(  ) {
  return function(d) { return "Computing the results of complex calculations can impact performance. Often, LODs are used on the fly to address issues of granularity in the data source that could be handled prior to analysis. Consider tailoring the data source to the analysis to avoid the need for as many LODs."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b850504208 = messageFormatterFn((function(  ) {
  return function(d) { return "Calculation length"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b442619817 = messageFormatterFn((function(  ) {
  return function(d) { return "The calculation " + d.calculationName + " in data source " + d.datasourceName + " is " + d.calculationLength + " characters long."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b271332729 = messageFormatterFn((function(  ) {
  return function(d) { return "Multiple calculations are more than 500 characters long."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1284389191 = messageFormatterFn((function(  ) {
  return function(d) { return "The workbook follows best practices and does not use long calculations."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b319006579 = messageFormatterFn((function(  ) {
  return function(d) { return "Computing the results of complex calculations can impact performance. Consider breaking the calculation apart and moving whatever is possible to the database, or using Tableau Prep to create the calculations prior to analysis."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1151374151 = messageFormatterFn((function(  ) {
  return function(d) { return "Unused data sources"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1477419950 = messageFormatterFn((function(  ) {
  return function(d) { return "The data source " + d.datasourceName + " isn't used in the workbook."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1512309026 = messageFormatterFn((function(  ) {
  return function(d) { return "Multiple data sources aren't used in the workbook."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b17459362 = messageFormatterFn((function(  ) {
  return function(d) { return "The workbook follows best practices and uses all the data sources."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a594722212 = messageFormatterFn((function(  ) {
  return function(d) { return "Each data source contributes to the amount of time Tableau spends loading and rendering a workbook. Consider closing data sources that aren't in use."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a198960398 = messageFormatterFn((function(  ) {
  return function(d) { return "Unused data source closed."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b947855298 = messageFormatterFn((function(  ) {
  return function(d) { return "Unused data sources closed."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a389404006 = messageFormatterFn((function(  ) {
  return function(d) { return "Number of joins"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b661672543 = messageFormatterFn((function(  ) {
  return function(d) { return "The data source " + d.datasourceName + " contains " + d.numJoins + " joins."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b885496367 = messageFormatterFn((function(  ) {
  return function(d) { return "Multiple data sources contain more than 5 joins."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1846877969 = messageFormatterFn((function(  ) {
  return function(d) { return "The workbook follows best practices and contains a limited number of joins."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1740461993 = messageFormatterFn((function(  ) {
  return function(d) { return "Data sources with many joins can make it hard to optimize queries. Consider using relationships where possible. Alternatively, multiple data sources with fewer, well-designed joins can perform better than a single, complex data source with many joins. This is especially true when there are many different levels of detail in a single data source."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b188080022 = messageFormatterFn((function(  ) {
  return function(d) { return "Uses data blending"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1455785757 = messageFormatterFn((function(  ) {
  return function(d) { return "The sheet " + d.sheetName + " uses data blending."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a397083469 = messageFormatterFn((function(  ) {
  return function(d) { return "Multiple sheets use data blending."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2018044813 = messageFormatterFn((function(  ) {
  return function(d) { return "The workbook follows best practices and does not use data blending."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1546080045 = messageFormatterFn((function(  ) {
  return function(d) { return "Data blending performance is driven by the number of unique members in the linking fields. Consider using relationships when possible. If a blend is required, try to use low cardinality linking fields."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1524929775 = messageFormatterFn((function(  ) {
  return function(d) { return "Non-materialized calculations"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1599428664 = messageFormatterFn((function(  ) {
  return function(d) { return "The data source " + d.datasourceName + " has not been pre-computed."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1559131800 = messageFormatterFn((function(  ) {
  return function(d) { return "Multiple data sources have not been pre-computed."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a608189544 = messageFormatterFn((function(  ) {
  return function(d) { return "The workbook follows best practices with pre-computed data sources."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a214961326 = messageFormatterFn((function(  ) {
  return function(d) { return "Computing the results of a calculation can impact performance. Consider pushing calculations to the data source when possible. If using an extract, materialize calculations to pre-compute their results."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b122135069 = messageFormatterFn((function(  ) {
  return function(d) { return "Number of layout containers"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b717069884 = messageFormatterFn((function(  ) {
  return function(d) { return "The dashboard " + d.dashboardName + " contains " + d.numLayoutContainers + " layout containers."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1498564340 = messageFormatterFn((function(  ) {
  return function(d) { return "Multiple dashboards contain more than 20 layout containers."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1167397644 = messageFormatterFn((function(  ) {
  return function(d) { return "The workbook follows best practices for the number of layout containers."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1186330310 = messageFormatterFn((function(  ) {
  return function(d) { return "Layout containers can complicate dashboard rendering. Consider removing unnecessary layout containers and simplifying the dashboard design."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1308841123 = messageFormatterFn((function(  ) {
  return function(d) { return "Dashboard size not fixed"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1660878076 = messageFormatterFn((function(  ) {
  return function(d) { return "The dashboard " + d.dashboardName + " is not fixed size."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1644451788 = messageFormatterFn((function(  ) {
  return function(d) { return "Multiple dashboards are not fixed size."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1081843252 = messageFormatterFn((function(  ) {
  return function(d) { return "The workbook follows best practices and uses fixed dashboard sizing."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a113244794 = messageFormatterFn((function(  ) {
  return function(d) { return "Fixed sized dashboards can be cached because they are a predictable size. Using automatic dashboard sizing means the results depend on the user's screen, and therefore the dashboard must be rendered every time. Rendering dashboards more often comes with a performance hit."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1314950844 = messageFormatterFn((function(  ) {
  return function(d) { return "Live data connections"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a873534731 = messageFormatterFn((function(  ) {
  return function(d) { return "The data source " + d.datasourceName + " is not an extract."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a45663035 = messageFormatterFn((function(  ) {
  return function(d) { return "Multiple data sources are not extracts."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b301871365 = messageFormatterFn((function(  ) {
  return function(d) { return "The workbook follows best practices and uses extracts."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2095768129 = messageFormatterFn((function(  ) {
  return function(d) { return "Tableau extracts are designed to be as efficient as possible for use with analytics. Using extracts is one of the easiest ways to improve performance due to the data source. Extracts also have many native features for optimization."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1021009899 = messageFormatterFn((function(  ) {
  return function(d) { return "Data source extracted."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a628764187 = messageFormatterFn((function(  ) {
  return function(d) { return "All data sources extracted."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b47452622 = messageFormatterFn((function(  ) {
  return function(d) { return "Uses grouping"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b730154411 = messageFormatterFn((function( plural, pluralFuncs ) {
  return function(d) { return plural(d.itemCount, 0, pluralFuncs.en, { one: function() { return "The data source " + d.datasourceName + " uses " + d.numGroupFields + " grouped field.";}, other: function() { return "The data source " + d.datasourceName + " uses " + d.numGroupFields + " grouped fields.";} }); }
})(messageFormat.plural, {"en": Globalize("en").pluralGenerator()}), Globalize("en").pluralGenerator({}));
Globalize.a561348229 = messageFormatterFn((function(  ) {
  return function(d) { return "Multiple data sources use grouped fields."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1972841275 = messageFormatterFn((function(  ) {
  return function(d) { return "The workbook follows best practices and uses minimal grouped fields."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a355553803 = messageFormatterFn((function(  ) {
  return function(d) { return "Tableau’s native grouping functionality loads the entire domain of the field, which can have a performance impact. Using a calculated field with a CASE statement to group instead can have better performance."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a868918454 = messageFormatterFn((function(  ) {
  return function(d) { return "Filter uses “Only Relevant Values”"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1541871697 = messageFormatterFn((function(  ) {
  return function(d) { return "The filter " + d.filterName + " on sheet " + d.sheetName + " uses “Only Relevant Values”."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1284644993 = messageFormatterFn((function(  ) {
  return function(d) { return "Multiple filters use “Only Relevant Values”."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b2030189119 = messageFormatterFn((function(  ) {
  return function(d) { return "The workbook follows best practices, avoiding “Only Relevant Values” filters."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1993670407 = messageFormatterFn((function(  ) {
  return function(d) { return "When a filter uses “Only Relevant Values”, the quick filter control only shows options that are applicable given the current state of the view. Every time a change is made to other filters, the list of values to be displayed must be re-queried, which has a performance impact. If the end-user benefit is valuable enough that this feature should be used, consider extracting the data and optimizing the extract."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1210953679 = messageFormatterFn((function(  ) {
  return function(d) { return "“Only Relevant Values” turned off for the filter."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b46696351 = messageFormatterFn((function(  ) {
  return function(d) { return "“Only Relevant Values” turned off for all filters."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b436924312 = messageFormatterFn((function(  ) {
  return function(d) { return "Uses date calculations"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a106280159 = messageFormatterFn((function( plural, pluralFuncs ) {
  return function(d) { return plural(d.itemCount, 0, pluralFuncs.en, { one: function() { return "The data source " + d.datasourceName + " contains " + d.numDateCalcs + " date calculation.";}, other: function() { return "The data source " + d.datasourceName + " contains " + d.numDateCalcs + " date calculations.";} }); }
})(messageFormat.plural, {"en": Globalize("en").pluralGenerator()}), Globalize("en").pluralGenerator({}));
Globalize.b1701163249 = messageFormatterFn((function(  ) {
  return function(d) { return "Multiple data sources contain date calculations."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b893632561 = messageFormatterFn((function(  ) {
  return function(d) { return "The workbook follows best practices and uses minimal date calculations."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a968714005 = messageFormatterFn((function(  ) {
  return function(d) { return "Date logic can be complicated. Minimize the amount of date calculations and conversion you have to do in Tableau. If necessary, use DATEPARSE and MAKEDATE before other methods and try to use built-in functions like DATEDIFF() when possible. If filtering on a date, use relative date filters or a continuous date filter instead of a discrete filter."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a153827172 = messageFormatterFn((function(  ) {
  return function(d) { return "Nested calculations"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a2130028899 = messageFormatterFn((function( plural, pluralFuncs ) {
  return function(d) { return plural(d.itemCount, 0, pluralFuncs.en, { one: function() { return "Data source " + d.datasourceName + " has " + d.numCalcs + " calculation that references another calculation.";}, other: function() { return "Data source " + d.datasourceName + " has " + d.numCalcs + " calculations that refer to another calculation.";} }); }
})(messageFormat.plural, {"en": Globalize("en").pluralGenerator()}), Globalize("en").pluralGenerator({}));
Globalize.b1320190061 = messageFormatterFn((function(  ) {
  return function(d) { return "Multiple calculations reference other calculations."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b951336621 = messageFormatterFn((function(  ) {
  return function(d) { return "The workbook follows best practices and doesn’t contain nested calculations."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b602362727 = messageFormatterFn((function(  ) {
  return function(d) { return "Nesting calculations can add complications and additional processing, especially with IF statements and other performance intensive functions. Consider pushing the calculations to the data source or materialize them in an extract."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b113741507 = messageFormatterFn((function(  ) {
  return function(d) { return "Filter uses conditional logic"; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b1174834262 = messageFormatterFn((function(  ) {
  return function(d) { return "The filter " + d.filterName + " on sheet " + d.sheetName + " uses conditional logic."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1474564826 = messageFormatterFn((function(  ) {
  return function(d) { return "Multiple filters use conditional logic."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.b894797222 = messageFormatterFn((function(  ) {
  return function(d) { return "The workbook follows best practices and has no conditional logic filters."; }
})(), Globalize("en").pluralGenerator({}));
Globalize.a1682463904 = messageFormatterFn((function(  ) {
  return function(d) { return "Dimension filtering can be done in a variety of ways, such as on a list of values, by wildcard matching, or using conditional logic. Conditional logic can be slow. Consider changing the filter to not use conditional logic. If necessary, make sure your logic tests for the most frequent outcome first and use ELSEIF or CASE when possible."; }
})(), Globalize("en").pluralGenerator({}));
}));
