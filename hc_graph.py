import ezhc as hc

"""
    simple definition of HighCharts template that are to be used in the Jupyter Notebook for advanced 
    financial tracks plotting
"""


def hc_stock(df_stocks, title=""):

    g = hc.Highstock()

    g.chart.height = 550
    g.legend.enabled = True
    g.legend.layout = 'horizontal'
    g.legend.align = 'center'
    g.legend.maxHeight = 100
    g.tooltip.enabled = True
    g.tooltip.valueDecimals = 2
    g.exporting.enabled = True

    g.chart.zoomType = 'xy'
    g.title.text = title

    g.plotOptions.series.compare = 'percent'
    g.yAxis.labels.formatter = hc.scripts.FORMATTER_PERCENT
    g.tooltip.pointFormat = hc.scripts.TOOLTIP_POINT_FORMAT_PERCENT
    g.tooltip.positioner = hc.scripts.TOOLTIP_POSITIONER_CENTER_TOP

    g.xAxis.gridLineWidth = 1.0
    g.xAxis.gridLineDashStyle = 'Dot'
    g.yAxis.gridLineWidth = 1.0
    g.yAxis.gridLineDashStyle = 'Dot'

    g.series = hc.build.series(df_stocks)
    return g


def hc_piechart(df, title=""):
    g = hc.Highcharts()

    g.chart.type = 'pie'
    g.chart.width = 800
    g.chart.height = 500
    g.exporting = False
    gpo = g.plotOptions.pie
    gpo.showInLegend = False
    gpo.dataLabels.enabled = True
    gpo.dataLabels.format = '{point.name}: {point.y:.1f}%'
    gpo.center = ['50%', '50%']
    gpo.size = '65%'
    g.drilldown.drillUpButton.position = {'x': 0, 'y': 0}
    g.tooltip.pointFormat = '<span style="color:{series.color}">{series.name}: <b>{point.y:,.3f}%</b><br/>'

    g.title.text = title

    g.series, g.drilldown.series = hc.build.series_drilldown(df)

    return g


def hc_spiderweb(df, title=""):
    g = hc.Highcharts()

    # g.chart.type = 'column'
    g.chart.polar = True
    g.plotOptions.series.animation = True

    g.chart.width = 800
    g.chart.height = 500
    g.pane.size = '90%'

    g.title.text = ""

    g.xAxis.type = 'category'
    g.xAxis.tickmarkPlacement = 'on'
    g.xAxis.lineWidth = 0

    g.yAxis.gridLineInterpolation = 'polygon'
    g.yAxis.lineWidth = 0
    g.yAxis.plotLines = [{'color': 'gray', 'value': 0, 'width': 1.5}]

    g.tooltip.pointFormat = '<span style="color:{series.color}">{series.name}: <b>{point.y:,.3f}%</b><br/>'
    g.tooltip.shared = True

    g.legend.enabled = True
    g.legend.align = 'right'
    g.legend.verticalAlign = 'top'
    g.legend.y = 70
    g.legend.layout = 'vertical'

    # color names from http://www.w3schools.com/colors/colors_names.asp
    # color rgba() codes from http://www.hexcolortool.com/
    g.series, g.drilldown.series = hc.build.series_drilldown(df, colorByPoint=False)

    return g
